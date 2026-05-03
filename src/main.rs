use rand::{rngs::OsRng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use ripemd::Ripemd160;
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use sha2::{Digest, Sha256};
use std::{
    env, fs, process,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "opencl-toy")]
use std::{
    ffi::CString,
    os::raw::{c_char, c_int, c_uint, c_void},
    ptr,
};

#[cfg(feature = "opencl-toy")]
const OPENCL_KEY_BENCHMARK_KERNEL: &str = include_str!("../opencl/key_benchmark.cl");

const EMBEDDED_WALLETS: &str = include_str!("../data/wallets.csv");
const DEFAULT_STATS_SECONDS: u64 = 5;
const DEFAULT_SUCCESS_FILE: &str = "satoshi-guesser-success.txt";
const DEFAULT_TOY_TARGET_NONCE: u64 = 500_000_000;
const DEFAULT_TOY_CUDA_BLOCKS: u32 = 1024;
const DEFAULT_TOY_CUDA_THREADS: u32 = 256;
const DEFAULT_TOY_CUDA_ITERS: u32 = 256;
const DEFAULT_TOY_OPENCL_GLOBAL_WORK_ITEMS: usize = 262_144;
const DEFAULT_TOY_OPENCL_LOCAL_WORK_ITEMS: usize = 256;
const DEFAULT_TOY_OPENCL_ITERS: u32 = 256;
const SYNTHETIC_OPENCL_TARGET_HASH160: [u32; 5] =
    [0x89abcdef, 0x01234567, 0xfedcba98, 0x76543210, 0x0badc0de];
const SATS_PER_BTC: u64 = 100_000_000;

#[derive(Clone, Debug)]
struct Args {
    threads: usize,
    stats_seconds: u64,
    targets_path: Option<String>,
    success_file: String,
    compressed: bool,
    uncompressed: bool,
    opencl_key_benchmark: bool,
    toy_gpu_demo: bool,
    toy_benchmark: bool,
    toy_target_nonce: u64,
    toy_cuda_blocks: u32,
    toy_cuda_threads: u32,
    toy_cuda_iters: u32,
    toy_opencl_global: usize,
    toy_opencl_local: usize,
    toy_opencl_iters: u32,
}

#[derive(Clone, Debug)]
struct Target {
    hash160: [u8; 20],
    balance_sats: u64,
    address: String,
}

#[derive(Debug)]
struct TargetSet {
    hashes: Vec<[u8; 20]>,
    targets: Vec<Target>,
}

impl TargetSet {
    fn len(&self) -> usize {
        self.hashes.len()
    }

    fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }
}

#[derive(Debug)]
struct Hit {
    thread_id: usize,
    private_key_hex: String,
    private_key_wif: String,
    address: String,
    balance_sats: u64,
    compressed: bool,
}

#[derive(Debug)]
struct ToyHit {
    source: String,
    nonce: u64,
    hash: u64,
}

#[derive(Clone, Debug)]
struct ToyOpenClDevice {
    platform_index: usize,
    device_index: usize,
    name: String,
}

fn main() {
    let args = parse_args().unwrap_or_else(|err| {
        eprintln!("{err}");
        eprintln!();
        print_usage_and_exit(2);
    });

    if args.opencl_key_benchmark {
        run_opencl_key_benchmark(args);
        return;
    }

    if args.toy_gpu_demo {
        run_toy_gpu_demo(args);
        return;
    }

    let csv = match &args.targets_path {
        Some(path) => fs::read_to_string(path).unwrap_or_else(|err| {
            eprintln!("failed to read targets file {path}: {err}");
            process::exit(1);
        }),
        None => EMBEDDED_WALLETS.to_owned(),
    };

    let targets = Arc::new(load_targets(&csv).unwrap_or_else(|err| {
        eprintln!("failed to load wallet targets: {err}");
        process::exit(1);
    }));

    if targets.is_empty() {
        eprintln!("no P2PKH wallet targets loaded");
        process::exit(1);
    }

    let total_balance_sats: u128 = targets.targets.iter().map(|t| t.balance_sats as u128).sum();
    eprintln!(
        "loaded {} targets ({:.8} BTC); starting {} worker threads; cpu_features={}",
        targets.len(),
        total_balance_sats as f64 / SATS_PER_BTC as f64,
        args.threads,
        cpu_feature_summary()
    );

    let total_guesses = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let (hit_tx, hit_rx) = mpsc::channel();
    let mut workers = Vec::with_capacity(args.threads);

    for thread_id in 0..args.threads {
        let worker_targets = Arc::clone(&targets);
        let worker_total = Arc::clone(&total_guesses);
        let worker_stop = Arc::clone(&stop);
        let worker_tx = hit_tx.clone();
        let worker_args = args.clone();
        workers.push(thread::spawn(move || {
            run_worker(
                thread_id,
                worker_targets,
                worker_total,
                worker_stop,
                worker_tx,
                worker_args.compressed,
                worker_args.uncompressed,
            );
        }));
    }
    drop(hit_tx);

    let started = Instant::now();
    let mut last_tick = started;
    let mut last_total = 0_u64;
    let interval = Duration::from_secs(args.stats_seconds);

    loop {
        match hit_rx.recv_timeout(interval) {
            Ok(hit) => {
                stop.store(true, AtomicOrdering::Relaxed);
                let total = total_guesses.load(AtomicOrdering::Relaxed);
                let elapsed = Instant::now().duration_since(started);
                let record = format_success_record(&hit, total, elapsed);
                match fs::write(&args.success_file, record.as_bytes()) {
                    Ok(()) => {
                        println!("success_file={}", args.success_file);
                    }
                    Err(err) => {
                        eprintln!("failed to write success file {}: {err}", args.success_file);
                    }
                }
                println!(
                    "HIT thread={} total_guesses={} elapsed_seconds={} compressed={} address={} balance_btc={:.8} private_key_hex={} private_key_wif={}",
                    hit.thread_id,
                    total,
                    elapsed.as_secs(),
                    hit.compressed,
                    hit.address,
                    hit.balance_sats as f64 / SATS_PER_BTC as f64,
                    hit.private_key_hex,
                    hit.private_key_wif
                );
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let now = Instant::now();
                let total = total_guesses.load(AtomicOrdering::Relaxed);
                let elapsed = now.duration_since(started).as_secs_f64();
                let tick_elapsed = now.duration_since(last_tick).as_secs_f64();
                let interval_guesses = total.saturating_sub(last_total);
                let interval_rate = interval_guesses as f64 / tick_elapsed;
                let average_rate = total as f64 / elapsed.max(f64::EPSILON);
                println!(
                    "stats elapsed={:.0}s total_guesses={} guesses_per_second={:.0} average_guesses_per_second={:.0}",
                    elapsed, total, interval_rate, average_rate
                );
                last_tick = now;
                last_total = total;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    stop.store(true, AtomicOrdering::Relaxed);
    for worker in workers {
        let _ = worker.join();
    }
}

fn run_opencl_key_benchmark(args: Args) {
    if !toy_opencl_compiled() {
        eprintln!(
            "opencl_key_benchmark_status=not_compiled build with: cargo build --release --features opencl-toy"
        );
        process::exit(1);
    }

    let opencl_devices = toy_opencl_devices();
    if opencl_devices.is_empty() {
        eprintln!("opencl_key_benchmark_status=no_opencl_gpu_devices_found");
        process::exit(1);
    }

    eprintln!(
        "opencl_key_benchmark target_hash160={} opencl_devices={} global_work_items={} local_work_items={} iterations_per_item={}",
        synthetic_opencl_target_hash160_hex(),
        opencl_devices.len(),
        args.toy_opencl_global,
        args.toy_opencl_local,
        args.toy_opencl_iters
    );
    for device in &opencl_devices {
        eprintln!(
            "opencl_key_device platform={} device={} name={}",
            device.platform_index, device.device_index, device.name
        );
    }

    let total_guesses = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let mut workers = Vec::with_capacity(opencl_devices.len());
    let total_lanes = opencl_devices.len() as u64;

    for (lane, device) in opencl_devices.into_iter().enumerate() {
        let worker_total = Arc::clone(&total_guesses);
        let worker_stop = Arc::clone(&stop);
        let global_work_items = args.toy_opencl_global;
        let local_work_items = args.toy_opencl_local;
        let iterations_per_item = args.toy_opencl_iters;
        workers.push(thread::spawn(move || {
            run_opencl_key_benchmark_worker(
                device,
                lane as u64,
                total_lanes,
                global_work_items,
                local_work_items,
                iterations_per_item,
                worker_total,
                worker_stop,
            );
        }));
    }

    let started = Instant::now();
    let mut last_tick = started;
    let mut last_total = 0_u64;
    let interval = Duration::from_secs(args.stats_seconds);

    loop {
        thread::sleep(interval);
        let now = Instant::now();
        let total = total_guesses.load(AtomicOrdering::Relaxed);
        let elapsed = now.duration_since(started).as_secs_f64();
        let tick_elapsed = now.duration_since(last_tick).as_secs_f64();
        let interval_guesses = total.saturating_sub(last_total);
        let interval_rate = interval_guesses as f64 / tick_elapsed;
        let average_rate = total as f64 / elapsed.max(f64::EPSILON);
        println!(
            "opencl_key_stats elapsed={:.0}s total_keys={} keys_per_second={:.0} average_keys_per_second={:.0}",
            elapsed, total, interval_rate, average_rate
        );
        last_tick = now;
        last_total = total;

        if workers.iter().all(|worker| worker.is_finished()) {
            eprintln!("opencl_key_benchmark_status=all_workers_stopped");
            break;
        }
    }

    stop.store(true, AtomicOrdering::Relaxed);
    for worker in workers {
        let _ = worker.join();
    }
}

fn run_toy_gpu_demo(args: Args) {
    let target_hash = toy_hash_u64(args.toy_target_nonce);
    let report_hits = !args.toy_benchmark;
    let cuda_devices = toy_cuda_device_count();
    let opencl_devices = toy_opencl_devices();
    let total_lanes = args.threads + cuda_devices + opencl_devices.len();

    eprintln!(
        "toy_gpu_demo mode={}; target_nonce={} target_hash={:#018x}; cpu_threads={}; cuda_devices={}; opencl_devices={}; cpu_features={}",
        if args.toy_benchmark { "benchmark" } else { "find" },
        args.toy_target_nonce,
        target_hash,
        args.threads,
        cuda_devices,
        opencl_devices.len(),
        cpu_feature_summary()
    );

    if cuda_devices > 0 {
        for device in 0..cuda_devices {
            eprintln!(
                "toy_cuda_device index={} name={}",
                device,
                toy_cuda_device_name(device)
            );
        }
    } else if !toy_cuda_compiled() {
        eprintln!(
            "toy_cuda_status=not_compiled build with: cargo build --release --features cuda-toy"
        );
    } else {
        eprintln!("toy_cuda_status=no_cuda_devices_found");
    }

    if !opencl_devices.is_empty() {
        for device in &opencl_devices {
            eprintln!(
                "toy_opencl_device platform={} device={} name={}",
                device.platform_index, device.device_index, device.name
            );
        }
    } else if !toy_opencl_compiled() {
        eprintln!(
            "toy_opencl_status=not_compiled build with: cargo build --release --features opencl-toy"
        );
    } else {
        eprintln!("toy_opencl_status=no_opencl_gpu_devices_found");
    }

    let total_guesses = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let (hit_tx, hit_rx) = mpsc::channel();
    let mut workers = Vec::with_capacity(total_lanes);

    for worker_id in 0..args.threads {
        let worker_total = Arc::clone(&total_guesses);
        let worker_stop = Arc::clone(&stop);
        let worker_tx = hit_tx.clone();
        workers.push(thread::spawn(move || {
            run_toy_cpu_worker(
                worker_id,
                worker_id as u64,
                total_lanes as u64,
                target_hash,
                report_hits,
                worker_total,
                worker_stop,
                worker_tx,
            );
        }));
    }

    for device in 0..cuda_devices {
        let worker_total = Arc::clone(&total_guesses);
        let worker_stop = Arc::clone(&stop);
        let worker_tx = hit_tx.clone();
        let lane = args.threads + device;
        let blocks = args.toy_cuda_blocks;
        let threads_per_block = args.toy_cuda_threads;
        let iterations_per_thread = args.toy_cuda_iters;
        workers.push(thread::spawn(move || {
            run_toy_cuda_worker(
                device,
                lane as u64,
                total_lanes as u64,
                target_hash,
                blocks,
                threads_per_block,
                iterations_per_thread,
                report_hits,
                worker_total,
                worker_stop,
                worker_tx,
            );
        }));
    }

    for (opencl_index, device) in opencl_devices.into_iter().enumerate() {
        let worker_total = Arc::clone(&total_guesses);
        let worker_stop = Arc::clone(&stop);
        let worker_tx = hit_tx.clone();
        let lane = args.threads + cuda_devices + opencl_index;
        let global_work_items = args.toy_opencl_global;
        let local_work_items = args.toy_opencl_local;
        let iterations_per_item = args.toy_opencl_iters;
        workers.push(thread::spawn(move || {
            run_toy_opencl_worker(
                device,
                lane as u64,
                total_lanes as u64,
                target_hash,
                global_work_items,
                local_work_items,
                iterations_per_item,
                report_hits,
                worker_total,
                worker_stop,
                worker_tx,
            );
        }));
    }
    drop(hit_tx);

    let started = Instant::now();
    let mut last_tick = started;
    let mut last_total = 0_u64;
    let interval = Duration::from_secs(args.stats_seconds);

    loop {
        match hit_rx.recv_timeout(interval) {
            Ok(hit) => {
                stop.store(true, AtomicOrdering::Relaxed);
                let total = total_guesses.load(AtomicOrdering::Relaxed);
                let elapsed = Instant::now().duration_since(started);
                let record = format!(
                    concat!(
                        "satoshi-guesser-toy-success\n",
                        "elapsed_seconds={}\n",
                        "source={}\n",
                        "total_guesses={}\n",
                        "target_hash={:#018x}\n",
                        "found_nonce={}\n"
                    ),
                    elapsed.as_secs(),
                    hit.source,
                    total,
                    hit.hash,
                    hit.nonce
                );
                match fs::write(&args.success_file, record.as_bytes()) {
                    Ok(()) => println!("success_file={}", args.success_file),
                    Err(err) => {
                        eprintln!("failed to write success file {}: {err}", args.success_file)
                    }
                }
                println!(
                    "TOY_HIT source={} total_guesses={} elapsed_seconds={} target_hash={:#018x} found_nonce={}",
                    hit.source,
                    total,
                    elapsed.as_secs(),
                    hit.hash,
                    hit.nonce
                );
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let now = Instant::now();
                let total = total_guesses.load(AtomicOrdering::Relaxed);
                let elapsed = now.duration_since(started).as_secs_f64();
                let tick_elapsed = now.duration_since(last_tick).as_secs_f64();
                let interval_guesses = total.saturating_sub(last_total);
                let interval_rate = interval_guesses as f64 / tick_elapsed;
                let average_rate = total as f64 / elapsed.max(f64::EPSILON);
                println!(
                    "toy_stats elapsed={:.0}s total_guesses={} guesses_per_second={:.0} average_guesses_per_second={:.0}",
                    elapsed, total, interval_rate, average_rate
                );
                last_tick = now;
                last_total = total;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    stop.store(true, AtomicOrdering::Relaxed);
    for worker in workers {
        let _ = worker.join();
    }
}

fn run_toy_cpu_worker(
    worker_id: usize,
    mut nonce: u64,
    stride: u64,
    target_hash: u64,
    report_hits: bool,
    total_guesses: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    hit_tx: mpsc::Sender<ToyHit>,
) {
    let mut local_count = 0_u64;
    while !stop.load(AtomicOrdering::Relaxed) {
        local_count += 1;
        let hash = toy_hash_u64(nonce);
        if report_hits && hash == target_hash {
            flush_count(&total_guesses, &mut local_count);
            report_toy_hit(&hit_tx, format!("cpu-{worker_id}"), nonce, hash, &stop);
            break;
        }
        nonce = nonce.wrapping_add(stride);
        if local_count >= 65_536 {
            flush_count(&total_guesses, &mut local_count);
        }
    }
    flush_count(&total_guesses, &mut local_count);
}

#[allow(clippy::too_many_arguments)]
fn run_toy_cuda_worker(
    device: usize,
    mut base: u64,
    stride: u64,
    target_hash: u64,
    blocks: u32,
    threads_per_block: u32,
    iterations_per_thread: u32,
    report_hits: bool,
    total_guesses: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    hit_tx: mpsc::Sender<ToyHit>,
) {
    while !stop.load(AtomicOrdering::Relaxed) {
        match toy_cuda_search_device(
            device,
            base,
            stride,
            target_hash,
            blocks,
            threads_per_block,
            iterations_per_thread,
        ) {
            Ok((Some(nonce), searched)) => {
                total_guesses.fetch_add(searched, AtomicOrdering::Relaxed);
                if report_hits {
                    report_toy_hit(&hit_tx, format!("cuda-{device}"), nonce, target_hash, &stop);
                    break;
                }
                base = base.wrapping_add(searched.wrapping_mul(stride));
            }
            Ok((None, searched)) => {
                total_guesses.fetch_add(searched, AtomicOrdering::Relaxed);
                base = base.wrapping_add(searched.wrapping_mul(stride));
            }
            Err(err) => {
                eprintln!("toy_cuda_error device={device} error={err}");
                break;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_toy_opencl_worker(
    device: ToyOpenClDevice,
    mut base: u64,
    stride: u64,
    target_hash: u64,
    global_work_items: usize,
    local_work_items: usize,
    iterations_per_item: u32,
    report_hits: bool,
    total_guesses: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    hit_tx: mpsc::Sender<ToyHit>,
) {
    let source = format!("opencl-{}-{}", device.platform_index, device.device_index);
    while !stop.load(AtomicOrdering::Relaxed) {
        match toy_opencl_search_device(
            &device,
            base,
            stride,
            target_hash,
            global_work_items,
            local_work_items,
            iterations_per_item,
            !report_hits,
        ) {
            Ok((Some(nonce), searched)) => {
                total_guesses.fetch_add(searched, AtomicOrdering::Relaxed);
                if report_hits {
                    report_toy_hit(&hit_tx, source.clone(), nonce, target_hash, &stop);
                    break;
                }
                base = base.wrapping_add(searched.wrapping_mul(stride));
            }
            Ok((None, searched)) => {
                total_guesses.fetch_add(searched, AtomicOrdering::Relaxed);
                base = base.wrapping_add(searched.wrapping_mul(stride));
            }
            Err(err) => {
                eprintln!(
                    "toy_opencl_error platform={} device={} error={err}",
                    device.platform_index, device.device_index
                );
                break;
            }
        }
    }
}

fn run_opencl_key_benchmark_worker(
    device: ToyOpenClDevice,
    mut base: u64,
    stride: u64,
    global_work_items: usize,
    local_work_items: usize,
    iterations_per_item: u32,
    total_guesses: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
) {
    while !stop.load(AtomicOrdering::Relaxed) {
        match opencl_key_benchmark_device(
            &device,
            base,
            stride,
            global_work_items,
            local_work_items,
            iterations_per_item,
        ) {
            Ok(searched) => {
                total_guesses.fetch_add(searched, AtomicOrdering::Relaxed);
                base = base.wrapping_add(searched.wrapping_mul(stride));
            }
            Err(err) => {
                eprintln!(
                    "opencl_key_error platform={} device={} error={err}",
                    device.platform_index, device.device_index
                );
                break;
            }
        }
    }
}

fn report_toy_hit(
    hit_tx: &mpsc::Sender<ToyHit>,
    source: String,
    nonce: u64,
    hash: u64,
    stop: &AtomicBool,
) {
    if stop
        .compare_exchange(
            false,
            true,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        )
        .is_ok()
    {
        let _ = hit_tx.send(ToyHit {
            source,
            nonce,
            hash,
        });
    }
}

fn run_worker(
    thread_id: usize,
    targets: Arc<TargetSet>,
    total_guesses: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    hit_tx: mpsc::Sender<Hit>,
    check_compressed: bool,
    check_uncompressed: bool,
) {
    let secp = Secp256k1::new();
    let mut seed = <ChaCha20Rng as SeedableRng>::Seed::default();
    OsRng.fill_bytes(&mut seed);
    let mut rng = ChaCha20Rng::from_seed(seed);
    let mut private_key = [0_u8; 32];
    let mut local_count = 0_u64;

    while !stop.load(AtomicOrdering::Relaxed) {
        rng.fill_bytes(&mut private_key);
        let Ok(secret_key) = SecretKey::from_slice(&private_key) else {
            continue;
        };

        let public_key = PublicKey::from_secret_key(&secp, &secret_key);
        local_count += 1;

        if check_compressed {
            let hash = hash160(&public_key.serialize());
            if let Some(target) = find_target(&targets, &hash) {
                flush_count(&total_guesses, &mut local_count);
                report_hit(&hit_tx, thread_id, &private_key, target, true, &stop);
                break;
            }
        }

        if check_uncompressed {
            let hash = hash160(&public_key.serialize_uncompressed());
            if let Some(target) = find_target(&targets, &hash) {
                flush_count(&total_guesses, &mut local_count);
                report_hit(&hit_tx, thread_id, &private_key, target, false, &stop);
                break;
            }
        }

        if local_count >= 4096 {
            flush_count(&total_guesses, &mut local_count);
        }
    }

    flush_count(&total_guesses, &mut local_count);
}

fn report_hit(
    hit_tx: &mpsc::Sender<Hit>,
    thread_id: usize,
    private_key: &[u8; 32],
    target: &Target,
    compressed: bool,
    stop: &AtomicBool,
) {
    if stop
        .compare_exchange(
            false,
            true,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        )
        .is_ok()
    {
        let _ = hit_tx.send(Hit {
            thread_id,
            private_key_hex: to_hex(private_key),
            private_key_wif: private_key_to_wif(private_key, compressed),
            address: target.address.clone(),
            balance_sats: target.balance_sats,
            compressed,
        });
    }
}

fn flush_count(total_guesses: &AtomicU64, local_count: &mut u64) {
    if *local_count > 0 {
        total_guesses.fetch_add(*local_count, AtomicOrdering::Relaxed);
        *local_count = 0;
    }
}

fn hash160(bytes: &[u8]) -> [u8; 20] {
    let sha = Sha256::digest(bytes);
    let ripe = Ripemd160::digest(sha);
    let mut out = [0_u8; 20];
    out.copy_from_slice(&ripe);
    out
}

fn toy_hash_u64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[cfg(feature = "cuda-toy")]
extern "C" {
    fn satoshi_toy_cuda_device_count() -> i32;
    fn satoshi_toy_cuda_device_name(
        device: i32,
        out: *mut std::os::raw::c_char,
        out_len: i32,
    ) -> i32;
    fn satoshi_toy_cuda_search_device(
        device: i32,
        base: u64,
        stride: u64,
        target_hash: u64,
        blocks: u32,
        threads_per_block: u32,
        iterations_per_thread: u32,
        host_found_nonce: *mut u64,
        host_searched: *mut u64,
    ) -> i32;
}

#[cfg(feature = "cuda-toy")]
fn toy_cuda_compiled() -> bool {
    true
}

#[cfg(not(feature = "cuda-toy"))]
fn toy_cuda_compiled() -> bool {
    false
}

#[cfg(feature = "cuda-toy")]
fn toy_cuda_device_count() -> usize {
    unsafe { satoshi_toy_cuda_device_count().max(0) as usize }
}

#[cfg(not(feature = "cuda-toy"))]
fn toy_cuda_device_count() -> usize {
    0
}

#[cfg(feature = "cuda-toy")]
fn toy_cuda_device_name(device: usize) -> String {
    let mut buf = [0_i8; 128];
    let rc =
        unsafe { satoshi_toy_cuda_device_name(device as i32, buf.as_mut_ptr(), buf.len() as i32) };
    if rc != 0 {
        return format!("unknown({rc})");
    }

    unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(not(feature = "cuda-toy"))]
fn toy_cuda_device_name(_device: usize) -> String {
    "not-compiled".to_owned()
}

#[cfg(feature = "cuda-toy")]
fn toy_cuda_search_device(
    device: usize,
    base: u64,
    stride: u64,
    target_hash: u64,
    blocks: u32,
    threads_per_block: u32,
    iterations_per_thread: u32,
) -> Result<(Option<u64>, u64), i32> {
    let mut found_nonce = 0_u64;
    let mut searched = 0_u64;
    let rc = unsafe {
        satoshi_toy_cuda_search_device(
            device as i32,
            base,
            stride,
            target_hash,
            blocks,
            threads_per_block,
            iterations_per_thread,
            &mut found_nonce,
            &mut searched,
        )
    };

    match rc {
        1 => Ok((Some(found_nonce), searched)),
        0 => Ok((None, searched)),
        err => Err(err),
    }
}

#[cfg(not(feature = "cuda-toy"))]
fn toy_cuda_search_device(
    _device: usize,
    _base: u64,
    _stride: u64,
    _target_hash: u64,
    _blocks: u32,
    _threads_per_block: u32,
    _iterations_per_thread: u32,
) -> Result<(Option<u64>, u64), i32> {
    Err(-999)
}

#[cfg(feature = "opencl-toy")]
type ClPlatformId = *mut c_void;
#[cfg(feature = "opencl-toy")]
type ClDeviceId = *mut c_void;
#[cfg(feature = "opencl-toy")]
type ClContext = *mut c_void;
#[cfg(feature = "opencl-toy")]
type ClCommandQueue = *mut c_void;
#[cfg(feature = "opencl-toy")]
type ClProgram = *mut c_void;
#[cfg(feature = "opencl-toy")]
type ClKernel = *mut c_void;
#[cfg(feature = "opencl-toy")]
type ClMem = *mut c_void;

#[cfg(feature = "opencl-toy")]
const CL_SUCCESS: c_int = 0;
#[cfg(feature = "opencl-toy")]
const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
#[cfg(feature = "opencl-toy")]
const CL_DEVICE_NAME: c_uint = 0x102b;
#[cfg(feature = "opencl-toy")]
const CL_PROGRAM_BUILD_LOG: c_uint = 0x1183;
#[cfg(feature = "opencl-toy")]
const CL_MEM_READ_WRITE: u64 = 1;
#[cfg(feature = "opencl-toy")]
const CL_TRUE: c_uint = 1;

#[cfg(feature = "opencl-toy")]
const OPENCL_TOY_KERNEL: &str = r#"
static ulong toy_hash_ulong(ulong x) {
    x += 0x9e3779b97f4a7c15UL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9UL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebUL;
    return x ^ (x >> 31);
}

__kernel void toy_search(
    ulong base,
    ulong stride,
    ulong target_hash,
    uint iterations_per_item,
    uint benchmark_mode,
    __global ulong* found_nonce,
    __global uint* found_flag,
    __global ulong* checksum_out
) {
    ulong gid = (ulong)get_global_id(0);
    ulong work_size = (ulong)get_global_size(0);

    if (benchmark_mode != 0u) {
        ulong acc = base ^ stride ^ target_hash ^ gid;
        for (uint i = 0; i < iterations_per_item; i++) {
            ulong nonce = base + (gid + ((ulong)i * work_size)) * stride;
            ulong hash = toy_hash_ulong(nonce);
            acc ^= hash + 0x9e3779b97f4a7c15UL + (acc << 6) + (acc >> 2);
        }
        checksum_out[gid] = acc;
        return;
    }

    for (uint i = 0; i < iterations_per_item; i++) {
        ulong nonce = base + (gid + ((ulong)i * work_size)) * stride;
        if (toy_hash_ulong(nonce) == target_hash) {
            *found_nonce = nonce;
            *found_flag = 1u;
            return;
        }
    }
}
"#;

#[cfg(feature = "opencl-toy")]
#[cfg_attr(target_os = "macos", link(name = "OpenCL", kind = "framework"))]
#[cfg_attr(not(target_os = "macos"), link(name = "OpenCL"))]
extern "C" {
    fn clGetPlatformIDs(
        num_entries: c_uint,
        platforms: *mut ClPlatformId,
        num_platforms: *mut c_uint,
    ) -> c_int;
    fn clGetDeviceIDs(
        platform: ClPlatformId,
        device_type: u64,
        num_entries: c_uint,
        devices: *mut ClDeviceId,
        num_devices: *mut c_uint,
    ) -> c_int;
    fn clGetDeviceInfo(
        device: ClDeviceId,
        param_name: c_uint,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> c_int;
    fn clCreateContext(
        properties: *const isize,
        num_devices: c_uint,
        devices: *const ClDeviceId,
        pfn_notify: Option<unsafe extern "C" fn(*const c_char, *const c_void, usize, *mut c_void)>,
        user_data: *mut c_void,
        errcode_ret: *mut c_int,
    ) -> ClContext;
    fn clCreateCommandQueue(
        context: ClContext,
        device: ClDeviceId,
        properties: u64,
        errcode_ret: *mut c_int,
    ) -> ClCommandQueue;
    fn clCreateProgramWithSource(
        context: ClContext,
        count: c_uint,
        strings: *const *const c_char,
        lengths: *const usize,
        errcode_ret: *mut c_int,
    ) -> ClProgram;
    fn clBuildProgram(
        program: ClProgram,
        num_devices: c_uint,
        device_list: *const ClDeviceId,
        options: *const c_char,
        pfn_notify: Option<unsafe extern "C" fn(ClProgram, *mut c_void)>,
        user_data: *mut c_void,
    ) -> c_int;
    fn clGetProgramBuildInfo(
        program: ClProgram,
        device: ClDeviceId,
        param_name: c_uint,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> c_int;
    fn clCreateKernel(
        program: ClProgram,
        kernel_name: *const c_char,
        errcode_ret: *mut c_int,
    ) -> ClKernel;
    fn clCreateBuffer(
        context: ClContext,
        flags: u64,
        size: usize,
        host_ptr: *mut c_void,
        errcode_ret: *mut c_int,
    ) -> ClMem;
    fn clEnqueueWriteBuffer(
        command_queue: ClCommandQueue,
        buffer: ClMem,
        blocking_write: c_uint,
        offset: usize,
        size: usize,
        ptr: *const c_void,
        num_events_in_wait_list: c_uint,
        event_wait_list: *const c_void,
        event: *mut c_void,
    ) -> c_int;
    fn clSetKernelArg(
        kernel: ClKernel,
        arg_index: c_uint,
        arg_size: usize,
        arg_value: *const c_void,
    ) -> c_int;
    fn clEnqueueNDRangeKernel(
        command_queue: ClCommandQueue,
        kernel: ClKernel,
        work_dim: c_uint,
        global_work_offset: *const usize,
        global_work_size: *const usize,
        local_work_size: *const usize,
        num_events_in_wait_list: c_uint,
        event_wait_list: *const c_void,
        event: *mut c_void,
    ) -> c_int;
    fn clFinish(command_queue: ClCommandQueue) -> c_int;
    fn clEnqueueReadBuffer(
        command_queue: ClCommandQueue,
        buffer: ClMem,
        blocking_read: c_uint,
        offset: usize,
        size: usize,
        ptr: *mut c_void,
        num_events_in_wait_list: c_uint,
        event_wait_list: *const c_void,
        event: *mut c_void,
    ) -> c_int;
    fn clReleaseMemObject(memobj: ClMem) -> c_int;
    fn clReleaseKernel(kernel: ClKernel) -> c_int;
    fn clReleaseProgram(program: ClProgram) -> c_int;
    fn clReleaseCommandQueue(command_queue: ClCommandQueue) -> c_int;
    fn clReleaseContext(context: ClContext) -> c_int;
}

#[cfg(feature = "opencl-toy")]
fn toy_opencl_compiled() -> bool {
    true
}

#[cfg(not(feature = "opencl-toy"))]
fn toy_opencl_compiled() -> bool {
    false
}

#[cfg(feature = "opencl-toy")]
fn toy_opencl_devices() -> Vec<ToyOpenClDevice> {
    let mut devices = Vec::new();
    let platforms = match opencl_platforms() {
        Ok(platforms) => platforms,
        Err(err) => {
            eprintln!("toy_opencl_platform_error={err}");
            return devices;
        }
    };

    for (platform_index, platform) in platforms.into_iter().enumerate() {
        let platform_devices = match opencl_gpu_devices(platform) {
            Ok(platform_devices) => platform_devices,
            Err(_) => continue,
        };
        for (device_index, device) in platform_devices.into_iter().enumerate() {
            devices.push(ToyOpenClDevice {
                platform_index,
                device_index,
                name: opencl_device_name(device),
            });
        }
    }

    devices
}

#[cfg(not(feature = "opencl-toy"))]
fn toy_opencl_devices() -> Vec<ToyOpenClDevice> {
    Vec::new()
}

#[cfg(feature = "opencl-toy")]
fn toy_opencl_search_device(
    device: &ToyOpenClDevice,
    base: u64,
    stride: u64,
    target_hash: u64,
    global_work_items: usize,
    local_work_items: usize,
    iterations_per_item: u32,
    benchmark_mode: bool,
) -> Result<(Option<u64>, u64), String> {
    let device_id = opencl_device_by_index(device.platform_index, device.device_index)?;
    let source = CString::new(OPENCL_TOY_KERNEL).expect("OpenCL source has no interior NUL");
    let kernel_name = CString::new("toy_search").expect("kernel name has no interior NUL");

    let mut status = CL_SUCCESS;
    let context = unsafe {
        clCreateContext(
            ptr::null(),
            1,
            &device_id,
            None,
            ptr::null_mut(),
            &mut status,
        )
    };
    if status != CL_SUCCESS || context.is_null() {
        return Err(format!("clCreateContext failed: {status}"));
    }

    let queue = unsafe { clCreateCommandQueue(context, device_id, 0, &mut status) };
    if status != CL_SUCCESS || queue.is_null() {
        unsafe {
            clReleaseContext(context);
        }
        return Err(format!("clCreateCommandQueue failed: {status}"));
    }

    let source_ptr = source.as_ptr();
    let source_len = OPENCL_TOY_KERNEL.len();
    let program =
        unsafe { clCreateProgramWithSource(context, 1, &source_ptr, &source_len, &mut status) };
    if status != CL_SUCCESS || program.is_null() {
        unsafe {
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
        return Err(format!("clCreateProgramWithSource failed: {status}"));
    }

    let build_status =
        unsafe { clBuildProgram(program, 1, &device_id, ptr::null(), None, ptr::null_mut()) };
    if build_status != CL_SUCCESS {
        let log = opencl_build_log(program, device_id);
        unsafe {
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
        return Err(format!("clBuildProgram failed: {build_status}: {log}"));
    }

    let kernel = unsafe { clCreateKernel(program, kernel_name.as_ptr(), &mut status) };
    if status != CL_SUCCESS || kernel.is_null() {
        unsafe {
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
        return Err(format!("clCreateKernel failed: {status}"));
    }

    let found_nonce_buf = unsafe {
        clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            std::mem::size_of::<u64>(),
            ptr::null_mut(),
            &mut status,
        )
    };
    if status != CL_SUCCESS || found_nonce_buf.is_null() {
        unsafe {
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
        return Err(format!("clCreateBuffer(found_nonce) failed: {status}"));
    }

    let found_flag_buf = unsafe {
        clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            std::mem::size_of::<c_uint>(),
            ptr::null_mut(),
            &mut status,
        )
    };
    if status != CL_SUCCESS || found_flag_buf.is_null() {
        unsafe {
            clReleaseMemObject(found_nonce_buf);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
        return Err(format!("clCreateBuffer(found_flag) failed: {status}"));
    }

    let checksum_buf = unsafe {
        clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            global_work_items.saturating_mul(std::mem::size_of::<u64>()),
            ptr::null_mut(),
            &mut status,
        )
    };
    if status != CL_SUCCESS || checksum_buf.is_null() {
        unsafe {
            clReleaseMemObject(found_flag_buf);
            clReleaseMemObject(found_nonce_buf);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
        return Err(format!("clCreateBuffer(checksum) failed: {status}"));
    }

    let zero_nonce = 0_u64;
    let zero_flag = 0_u32;
    let mut found_nonce = 0_u64;
    let mut found_flag = 0_u32;
    let mut checksum_sample = 0_u64;
    let benchmark_mode_flag = u32::from(benchmark_mode);
    let search_result = unsafe {
        check_opencl(
            clEnqueueWriteBuffer(
                queue,
                found_nonce_buf,
                CL_TRUE,
                0,
                std::mem::size_of::<u64>(),
                &zero_nonce as *const u64 as *const c_void,
                0,
                ptr::null(),
                ptr::null_mut(),
            ),
            "clEnqueueWriteBuffer(found_nonce)",
        )?;
        check_opencl(
            clEnqueueWriteBuffer(
                queue,
                found_flag_buf,
                CL_TRUE,
                0,
                std::mem::size_of::<c_uint>(),
                &zero_flag as *const u32 as *const c_void,
                0,
                ptr::null(),
                ptr::null_mut(),
            ),
            "clEnqueueWriteBuffer(found_flag)",
        )?;

        set_opencl_arg(kernel, 0, &base)?;
        set_opencl_arg(kernel, 1, &stride)?;
        set_opencl_arg(kernel, 2, &target_hash)?;
        set_opencl_arg(kernel, 3, &iterations_per_item)?;
        set_opencl_arg(kernel, 4, &benchmark_mode_flag)?;
        set_opencl_arg(kernel, 5, &found_nonce_buf)?;
        set_opencl_arg(kernel, 6, &found_flag_buf)?;
        set_opencl_arg(kernel, 7, &checksum_buf)?;

        let global = [global_work_items];
        let local = [local_work_items];
        check_opencl(
            clEnqueueNDRangeKernel(
                queue,
                kernel,
                1,
                ptr::null(),
                global.as_ptr(),
                local.as_ptr(),
                0,
                ptr::null(),
                ptr::null_mut(),
            ),
            "clEnqueueNDRangeKernel",
        )?;
        check_opencl(clFinish(queue), "clFinish")?;

        if benchmark_mode {
            check_opencl(
                clEnqueueReadBuffer(
                    queue,
                    checksum_buf,
                    CL_TRUE,
                    0,
                    std::mem::size_of::<u64>(),
                    &mut checksum_sample as *mut u64 as *mut c_void,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                ),
                "clEnqueueReadBuffer(checksum)",
            )
        } else {
            check_opencl(
                clEnqueueReadBuffer(
                    queue,
                    found_nonce_buf,
                    CL_TRUE,
                    0,
                    std::mem::size_of::<u64>(),
                    &mut found_nonce as *mut u64 as *mut c_void,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                ),
                "clEnqueueReadBuffer(found_nonce)",
            )?;
            check_opencl(
                clEnqueueReadBuffer(
                    queue,
                    found_flag_buf,
                    CL_TRUE,
                    0,
                    std::mem::size_of::<c_uint>(),
                    &mut found_flag as *mut u32 as *mut c_void,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                ),
                "clEnqueueReadBuffer(found_flag)",
            )
        }
    };

    unsafe {
        clReleaseMemObject(checksum_buf);
        clReleaseMemObject(found_flag_buf);
        clReleaseMemObject(found_nonce_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
    search_result?;
    std::hint::black_box(checksum_sample);

    let searched = (global_work_items as u64).saturating_mul(iterations_per_item as u64);
    if found_flag != 0 {
        Ok((Some(found_nonce), searched))
    } else {
        Ok((None, searched))
    }
}

#[cfg(not(feature = "opencl-toy"))]
fn toy_opencl_search_device(
    _device: &ToyOpenClDevice,
    _base: u64,
    _stride: u64,
    _target_hash: u64,
    _global_work_items: usize,
    _local_work_items: usize,
    _iterations_per_item: u32,
    _benchmark_mode: bool,
) -> Result<(Option<u64>, u64), String> {
    Err("opencl-toy feature is not compiled".to_owned())
}

#[cfg(feature = "opencl-toy")]
fn opencl_key_benchmark_device(
    device: &ToyOpenClDevice,
    base: u64,
    stride: u64,
    global_work_items: usize,
    local_work_items: usize,
    iterations_per_item: u32,
) -> Result<u64, String> {
    let device_id = opencl_device_by_index(device.platform_index, device.device_index)?;
    let source =
        CString::new(OPENCL_KEY_BENCHMARK_KERNEL).expect("OpenCL source has no interior NUL");
    let kernel_name = CString::new("key_benchmark").expect("kernel name has no interior NUL");

    let mut status = CL_SUCCESS;
    let context = unsafe {
        clCreateContext(
            ptr::null(),
            1,
            &device_id,
            None,
            ptr::null_mut(),
            &mut status,
        )
    };
    if status != CL_SUCCESS || context.is_null() {
        return Err(format!("clCreateContext failed: {status}"));
    }

    let queue = unsafe { clCreateCommandQueue(context, device_id, 0, &mut status) };
    if status != CL_SUCCESS || queue.is_null() {
        unsafe {
            clReleaseContext(context);
        }
        return Err(format!("clCreateCommandQueue failed: {status}"));
    }

    let source_ptr = source.as_ptr();
    let source_len = OPENCL_KEY_BENCHMARK_KERNEL.len();
    let program =
        unsafe { clCreateProgramWithSource(context, 1, &source_ptr, &source_len, &mut status) };
    if status != CL_SUCCESS || program.is_null() {
        unsafe {
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
        return Err(format!("clCreateProgramWithSource failed: {status}"));
    }

    let build_status =
        unsafe { clBuildProgram(program, 1, &device_id, ptr::null(), None, ptr::null_mut()) };
    if build_status != CL_SUCCESS {
        let log = opencl_build_log(program, device_id);
        unsafe {
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
        return Err(format!("clBuildProgram failed: {build_status}: {log}"));
    }

    let kernel = unsafe { clCreateKernel(program, kernel_name.as_ptr(), &mut status) };
    if status != CL_SUCCESS || kernel.is_null() {
        unsafe {
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
        return Err(format!("clCreateKernel failed: {status}"));
    }

    let checksum_buf = unsafe {
        clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            global_work_items.saturating_mul(std::mem::size_of::<u64>()),
            ptr::null_mut(),
            &mut status,
        )
    };
    if status != CL_SUCCESS || checksum_buf.is_null() {
        unsafe {
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
        return Err(format!("clCreateBuffer(checksum) failed: {status}"));
    }

    let target = SYNTHETIC_OPENCL_TARGET_HASH160;
    let mut checksum_sample = 0_u64;
    let benchmark_result = unsafe {
        set_opencl_arg(kernel, 0, &base)?;
        set_opencl_arg(kernel, 1, &stride)?;
        set_opencl_arg(kernel, 2, &target[0])?;
        set_opencl_arg(kernel, 3, &target[1])?;
        set_opencl_arg(kernel, 4, &target[2])?;
        set_opencl_arg(kernel, 5, &target[3])?;
        set_opencl_arg(kernel, 6, &target[4])?;
        set_opencl_arg(kernel, 7, &iterations_per_item)?;
        set_opencl_arg(kernel, 8, &checksum_buf)?;

        let global = [global_work_items];
        let local = [local_work_items];
        check_opencl(
            clEnqueueNDRangeKernel(
                queue,
                kernel,
                1,
                ptr::null(),
                global.as_ptr(),
                local.as_ptr(),
                0,
                ptr::null(),
                ptr::null_mut(),
            ),
            "clEnqueueNDRangeKernel",
        )?;
        check_opencl(clFinish(queue), "clFinish")?;
        check_opencl(
            clEnqueueReadBuffer(
                queue,
                checksum_buf,
                CL_TRUE,
                0,
                std::mem::size_of::<u64>(),
                &mut checksum_sample as *mut u64 as *mut c_void,
                0,
                ptr::null(),
                ptr::null_mut(),
            ),
            "clEnqueueReadBuffer(checksum)",
        )
    };

    unsafe {
        clReleaseMemObject(checksum_buf);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
    benchmark_result?;
    std::hint::black_box(checksum_sample);

    Ok((global_work_items as u64).saturating_mul(iterations_per_item as u64))
}

#[cfg(not(feature = "opencl-toy"))]
fn opencl_key_benchmark_device(
    _device: &ToyOpenClDevice,
    _base: u64,
    _stride: u64,
    _global_work_items: usize,
    _local_work_items: usize,
    _iterations_per_item: u32,
) -> Result<u64, String> {
    Err("opencl-toy feature is not compiled".to_owned())
}

#[cfg(feature = "opencl-toy")]
fn opencl_platforms() -> Result<Vec<ClPlatformId>, String> {
    let mut count: c_uint = 0;
    let rc = unsafe { clGetPlatformIDs(0, ptr::null_mut(), &mut count) };
    if rc != CL_SUCCESS {
        return Err(format!("clGetPlatformIDs(count) failed: {rc}"));
    }
    if count == 0 {
        return Ok(Vec::new());
    }

    let mut platforms = vec![ptr::null_mut(); count as usize];
    let rc = unsafe { clGetPlatformIDs(count, platforms.as_mut_ptr(), ptr::null_mut()) };
    if rc != CL_SUCCESS {
        return Err(format!("clGetPlatformIDs(list) failed: {rc}"));
    }
    Ok(platforms)
}

#[cfg(feature = "opencl-toy")]
fn opencl_gpu_devices(platform: ClPlatformId) -> Result<Vec<ClDeviceId>, String> {
    let mut count: c_uint = 0;
    let rc =
        unsafe { clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, ptr::null_mut(), &mut count) };
    if rc != CL_SUCCESS || count == 0 {
        return Ok(Vec::new());
    }

    let mut devices = vec![ptr::null_mut(); count as usize];
    let rc = unsafe {
        clGetDeviceIDs(
            platform,
            CL_DEVICE_TYPE_GPU,
            count,
            devices.as_mut_ptr(),
            ptr::null_mut(),
        )
    };
    if rc != CL_SUCCESS {
        return Err(format!("clGetDeviceIDs(list) failed: {rc}"));
    }
    Ok(devices)
}

#[cfg(feature = "opencl-toy")]
fn opencl_device_by_index(
    platform_index: usize,
    device_index: usize,
) -> Result<ClDeviceId, String> {
    let platforms = opencl_platforms()?;
    let platform = platforms
        .get(platform_index)
        .copied()
        .ok_or_else(|| format!("OpenCL platform index {platform_index} no longer exists"))?;
    let devices = opencl_gpu_devices(platform)?;
    devices
        .get(device_index)
        .copied()
        .ok_or_else(|| format!("OpenCL device index {device_index} no longer exists"))
}

#[cfg(feature = "opencl-toy")]
fn opencl_device_name(device: ClDeviceId) -> String {
    let mut size = 0_usize;
    let rc = unsafe { clGetDeviceInfo(device, CL_DEVICE_NAME, 0, ptr::null_mut(), &mut size) };
    if rc != CL_SUCCESS || size == 0 {
        return format!("unknown({rc})");
    }

    let mut buf = vec![0_u8; size];
    let rc = unsafe {
        clGetDeviceInfo(
            device,
            CL_DEVICE_NAME,
            buf.len(),
            buf.as_mut_ptr() as *mut c_void,
            ptr::null_mut(),
        )
    };
    if rc != CL_SUCCESS {
        return format!("unknown({rc})");
    }

    if let Some(nul) = buf.iter().position(|&byte| byte == 0) {
        buf.truncate(nul);
    }
    String::from_utf8_lossy(&buf).into_owned()
}

#[cfg(feature = "opencl-toy")]
fn opencl_build_log(program: ClProgram, device: ClDeviceId) -> String {
    let mut size = 0_usize;
    let rc = unsafe {
        clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            0,
            ptr::null_mut(),
            &mut size,
        )
    };
    if rc != CL_SUCCESS || size == 0 {
        return format!("no build log ({rc})");
    }

    let mut buf = vec![0_u8; size];
    let rc = unsafe {
        clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            buf.len(),
            buf.as_mut_ptr() as *mut c_void,
            ptr::null_mut(),
        )
    };
    if rc != CL_SUCCESS {
        return format!("failed to read build log ({rc})");
    }

    if let Some(nul) = buf.iter().position(|&byte| byte == 0) {
        buf.truncate(nul);
    }
    String::from_utf8_lossy(&buf).into_owned()
}

#[cfg(feature = "opencl-toy")]
unsafe fn set_opencl_arg<T>(kernel: ClKernel, index: c_uint, value: &T) -> Result<(), String> {
    check_opencl(
        clSetKernelArg(
            kernel,
            index,
            std::mem::size_of::<T>(),
            value as *const T as *const c_void,
        ),
        "clSetKernelArg",
    )
}

#[cfg(feature = "opencl-toy")]
fn check_opencl(rc: c_int, name: &str) -> Result<(), String> {
    if rc == CL_SUCCESS {
        Ok(())
    } else {
        Err(format!("{name} failed: {rc}"))
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpu_feature_summary() -> String {
    let mut features = Vec::new();
    if std::is_x86_feature_detected!("sse2") {
        features.push("sse2");
    }
    if std::is_x86_feature_detected!("ssse3") {
        features.push("ssse3");
    }
    if std::is_x86_feature_detected!("sse4.1") {
        features.push("sse4.1");
    }
    if std::is_x86_feature_detected!("avx2") {
        features.push("avx2");
    }
    if std::is_x86_feature_detected!("sha") {
        features.push("sha-ni");
    }

    if features.is_empty() {
        "none-detected".to_owned()
    } else {
        features.join(",")
    }
}

#[cfg(target_arch = "aarch64")]
fn cpu_feature_summary() -> String {
    let mut features = Vec::new();
    if cfg!(target_vendor = "apple") && cfg!(target_os = "macos") {
        features.push("apple-silicon");
    }
    if std::arch::is_aarch64_feature_detected!("neon") {
        features.push("neon");
    }
    if std::arch::is_aarch64_feature_detected!("aes") {
        features.push("aes");
    }
    if std::arch::is_aarch64_feature_detected!("sha2") {
        features.push("sha2");
    }
    if std::arch::is_aarch64_feature_detected!("sha3") {
        features.push("sha3");
    }
    if std::arch::is_aarch64_feature_detected!("crc") {
        features.push("crc");
    }
    if std::arch::is_aarch64_feature_detected!("lse") {
        features.push("lse");
    }
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        features.push("dotprod");
    }
    if std::arch::is_aarch64_feature_detected!("fp16") {
        features.push("fp16");
    }

    if features.is_empty() {
        "none-detected".to_owned()
    } else {
        features.join(",")
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
fn cpu_feature_summary() -> String {
    "not-reported-for-this-arch".to_owned()
}

fn private_key_to_wif(private_key: &[u8; 32], compressed: bool) -> String {
    let mut payload = Vec::with_capacity(if compressed { 34 } else { 33 });
    payload.push(0x80);
    payload.extend_from_slice(private_key);
    if compressed {
        payload.push(0x01);
    }
    bs58::encode(payload).with_check().into_string()
}

fn format_success_record(hit: &Hit, total_guesses: u64, elapsed: Duration) -> String {
    let found_unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);

    format!(
        concat!(
            "satoshi-guesser-success\n",
            "found_unix_seconds={}\n",
            "elapsed_seconds={}\n",
            "thread={}\n",
            "total_guesses={}\n",
            "compressed={}\n",
            "address={}\n",
            "balance_sats={}\n",
            "balance_btc={:.8}\n",
            "private_key_hex={}\n",
            "private_key_wif={}\n"
        ),
        found_unix_seconds,
        elapsed.as_secs(),
        hit.thread_id,
        total_guesses,
        hit.compressed,
        hit.address,
        hit.balance_sats,
        hit.balance_sats as f64 / SATS_PER_BTC as f64,
        hit.private_key_hex,
        hit.private_key_wif
    )
}

fn find_target<'a>(targets: &'a TargetSet, hash: &[u8; 20]) -> Option<&'a Target> {
    targets
        .hashes
        .binary_search(hash)
        .ok()
        .map(|idx| &targets.targets[idx])
}

fn load_targets(csv: &str) -> Result<TargetSet, String> {
    let mut targets = Vec::new();

    for (line_no, line) in csv.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.to_ascii_lowercase().starts_with("address,") {
            continue;
        }

        let mut cols = trimmed.split(',');
        let address = cols
            .next()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("line {} is missing an address", line_no + 1))?;
        let balance_sats = cols
            .next()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(btc_to_sats)
            .transpose()?
            .unwrap_or(50 * SATS_PER_BTC);

        if let Some(hash160) = decode_p2pkh_hash160(address)? {
            targets.push(Target {
                hash160,
                balance_sats,
                address: address.to_owned(),
            });
        }
    }

    targets.sort_by(|a, b| a.hash160.cmp(&b.hash160));
    targets.dedup_by(|a, b| a.hash160 == b.hash160);
    let hashes = targets.iter().map(|target| target.hash160).collect();
    Ok(TargetSet { hashes, targets })
}

fn decode_p2pkh_hash160(address: &str) -> Result<Option<[u8; 20]>, String> {
    let payload = bs58::decode(address)
        .with_check(None)
        .into_vec()
        .map_err(|err| format!("invalid base58check address {address}: {err}"))?;

    if payload.len() != 21 || payload[0] != 0x00 {
        return Ok(None);
    }

    let mut hash = [0_u8; 20];
    hash.copy_from_slice(&payload[1..]);
    Ok(Some(hash))
}

fn btc_to_sats(input: &str) -> Result<u64, String> {
    let mut parts = input.split('.');
    let whole = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("invalid BTC amount {input}"))?;
    let frac = parts.next().unwrap_or("");
    if parts.next().is_some() || frac.len() > 8 {
        return Err(format!("invalid BTC amount {input}"));
    }

    let whole_sats = whole
        .parse::<u64>()
        .map_err(|_| format!("invalid BTC amount {input}"))?
        .checked_mul(SATS_PER_BTC)
        .ok_or_else(|| format!("BTC amount overflows u64: {input}"))?;

    let mut frac_padded = frac.to_owned();
    while frac_padded.len() < 8 {
        frac_padded.push('0');
    }
    let frac_sats = if frac_padded.is_empty() {
        0
    } else {
        frac_padded
            .parse::<u64>()
            .map_err(|_| format!("invalid BTC amount {input}"))?
    };

    whole_sats
        .checked_add(frac_sats)
        .ok_or_else(|| format!("BTC amount overflows u64: {input}"))
}

fn parse_args() -> Result<Args, String> {
    let mut threads = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let mut stats_seconds = DEFAULT_STATS_SECONDS;
    let mut targets_path = None;
    let mut success_file = DEFAULT_SUCCESS_FILE.to_owned();
    let mut compressed = true;
    let mut uncompressed = true;
    let mut opencl_key_benchmark = false;
    let mut toy_gpu_demo = false;
    let mut toy_benchmark = false;
    let mut toy_target_nonce = DEFAULT_TOY_TARGET_NONCE;
    let mut toy_cuda_blocks = DEFAULT_TOY_CUDA_BLOCKS;
    let mut toy_cuda_threads = DEFAULT_TOY_CUDA_THREADS;
    let mut toy_cuda_iters = DEFAULT_TOY_CUDA_ITERS;
    let mut toy_opencl_global = DEFAULT_TOY_OPENCL_GLOBAL_WORK_ITEMS;
    let mut toy_opencl_local = DEFAULT_TOY_OPENCL_LOCAL_WORK_ITEMS;
    let mut toy_opencl_iters = DEFAULT_TOY_OPENCL_ITERS;

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => print_usage_and_exit(0),
            "-t" | "--threads" => {
                threads = next_arg(&mut iter, "--threads")?
                    .parse::<usize>()
                    .map_err(|_| "--threads must be a positive integer".to_owned())?;
                if threads == 0 {
                    return Err("--threads must be greater than zero".to_owned());
                }
            }
            "--stats-seconds" => {
                stats_seconds = next_arg(&mut iter, "--stats-seconds")?
                    .parse::<u64>()
                    .map_err(|_| "--stats-seconds must be a positive integer".to_owned())?;
                if stats_seconds == 0 {
                    return Err("--stats-seconds must be greater than zero".to_owned());
                }
            }
            "--targets" => {
                targets_path = Some(next_arg(&mut iter, "--targets")?);
            }
            "--success-file" => {
                success_file = next_arg(&mut iter, "--success-file")?;
            }
            "--compressed-only" => {
                compressed = true;
                uncompressed = false;
            }
            "--uncompressed-only" => {
                compressed = false;
                uncompressed = true;
            }
            "--opencl-key-benchmark" => {
                opencl_key_benchmark = true;
            }
            "--toy-gpu-demo" => {
                toy_gpu_demo = true;
            }
            "--toy-benchmark" => {
                toy_gpu_demo = true;
                toy_benchmark = true;
            }
            "--toy-target-nonce" => {
                toy_target_nonce = next_arg(&mut iter, "--toy-target-nonce")?
                    .parse::<u64>()
                    .map_err(|_| "--toy-target-nonce must be a u64 integer".to_owned())?;
            }
            "--toy-cuda-blocks" => {
                toy_cuda_blocks = next_arg(&mut iter, "--toy-cuda-blocks")?
                    .parse::<u32>()
                    .map_err(|_| "--toy-cuda-blocks must be a positive integer".to_owned())?;
                if toy_cuda_blocks == 0 {
                    return Err("--toy-cuda-blocks must be greater than zero".to_owned());
                }
            }
            "--toy-cuda-threads" => {
                toy_cuda_threads = next_arg(&mut iter, "--toy-cuda-threads")?
                    .parse::<u32>()
                    .map_err(|_| "--toy-cuda-threads must be a positive integer".to_owned())?;
                if toy_cuda_threads == 0 {
                    return Err("--toy-cuda-threads must be greater than zero".to_owned());
                }
            }
            "--toy-cuda-iters" => {
                toy_cuda_iters = next_arg(&mut iter, "--toy-cuda-iters")?
                    .parse::<u32>()
                    .map_err(|_| "--toy-cuda-iters must be a positive integer".to_owned())?;
                if toy_cuda_iters == 0 {
                    return Err("--toy-cuda-iters must be greater than zero".to_owned());
                }
            }
            "--toy-opencl-global" => {
                toy_opencl_global = next_arg(&mut iter, "--toy-opencl-global")?
                    .parse::<usize>()
                    .map_err(|_| "--toy-opencl-global must be a positive integer".to_owned())?;
                if toy_opencl_global == 0 {
                    return Err("--toy-opencl-global must be greater than zero".to_owned());
                }
            }
            "--toy-opencl-local" => {
                toy_opencl_local = next_arg(&mut iter, "--toy-opencl-local")?
                    .parse::<usize>()
                    .map_err(|_| "--toy-opencl-local must be a positive integer".to_owned())?;
                if toy_opencl_local == 0 {
                    return Err("--toy-opencl-local must be greater than zero".to_owned());
                }
            }
            "--toy-opencl-iters" => {
                toy_opencl_iters = next_arg(&mut iter, "--toy-opencl-iters")?
                    .parse::<u32>()
                    .map_err(|_| "--toy-opencl-iters must be a positive integer".to_owned())?;
                if toy_opencl_iters == 0 {
                    return Err("--toy-opencl-iters must be greater than zero".to_owned());
                }
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    if toy_opencl_local > toy_opencl_global {
        return Err(
            "--toy-opencl-local must be less than or equal to --toy-opencl-global".to_owned(),
        );
    }
    if toy_opencl_global % toy_opencl_local != 0 {
        return Err("--toy-opencl-global must be a multiple of --toy-opencl-local".to_owned());
    }

    Ok(Args {
        threads,
        stats_seconds,
        targets_path,
        success_file,
        compressed,
        uncompressed,
        opencl_key_benchmark,
        toy_gpu_demo,
        toy_benchmark,
        toy_target_nonce,
        toy_cuda_blocks,
        toy_cuda_threads,
        toy_cuda_iters,
        toy_opencl_global,
        toy_opencl_local,
        toy_opencl_iters,
    })
}

fn next_arg(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn synthetic_opencl_target_hash160_hex() -> String {
    let mut bytes = [0_u8; 20];
    for (word_index, word) in SYNTHETIC_OPENCL_TARGET_HASH160.iter().enumerate() {
        let start = word_index * 4;
        bytes[start] = (*word & 0xff) as u8;
        bytes[start + 1] = ((*word >> 8) & 0xff) as u8;
        bytes[start + 2] = ((*word >> 16) & 0xff) as u8;
        bytes[start + 3] = ((*word >> 24) & 0xff) as u8;
    }
    to_hex(&bytes)
}

fn print_usage_and_exit(code: i32) -> ! {
    eprintln!(
        "Usage: satoshi-guesser [--threads N] [--stats-seconds N] [--targets wallets.csv] [--success-file path] [--compressed-only|--uncompressed-only] [--opencl-key-benchmark] [--toy-gpu-demo] [--toy-benchmark] [--toy-target-nonce N] [--toy-cuda-blocks N] [--toy-cuda-threads N] [--toy-cuda-iters N] [--toy-opencl-global N] [--toy-opencl-local N] [--toy-opencl-iters N]"
    );
    process::exit(code);
}

fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_genesis_address() {
        let hash = decode_p2pkh_hash160("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")
            .unwrap()
            .unwrap();
        assert_eq!(to_hex(&hash), "62e907b15cbf27d5425399ebf6f0fb50ebb88f18");
    }

    #[test]
    fn parses_btc_to_sats() {
        assert_eq!(btc_to_sats("50").unwrap(), 5_000_000_000);
        assert_eq!(btc_to_sats("50.00000001").unwrap(), 5_000_000_001);
    }

    #[test]
    fn embedded_targets_include_genesis() {
        let targets = load_targets(EMBEDDED_WALLETS).unwrap();
        let genesis = decode_p2pkh_hash160("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")
            .unwrap()
            .unwrap();
        assert!(find_target(&targets, &genesis).is_some());
    }

    #[test]
    fn known_private_key_derives_expected_uncompressed_hash160() {
        let private_key =
            from_hex_32("18e14a7b6a307f426a94f8114701e7c8e774e7f9a47e2c2035db29a206321725");
        let secp = Secp256k1::new();
        let secret_key = SecretKey::from_slice(&private_key).unwrap();
        let public_key = PublicKey::from_secret_key(&secp, &secret_key);
        assert_eq!(
            to_hex(&hash160(&public_key.serialize_uncompressed())),
            "010966776006953d5567439e5e39f86a0d273bee"
        );
    }

    #[test]
    fn encodes_private_key_to_wif() {
        let private_key =
            from_hex_32("0c28fca386c7a227600b2fe50b7cae11ec86d3bf1fbe471be89827e19d72aa1d");
        assert_eq!(
            private_key_to_wif(&private_key, false),
            "5HueCGU8rMjxEXxiPuD5BDku4MkFqeZyd4dZ1jvhTVqvbTLvyTJ"
        );
        assert_eq!(
            private_key_to_wif(&private_key, true),
            "KwdMAjGmerYanjeui5SHS7JkmpZvVipYvB2LJGU1ZxJwYvP98617"
        );
    }

    #[test]
    fn success_record_contains_key_material_and_match() {
        let hit = Hit {
            thread_id: 7,
            private_key_hex: "0c28fca386c7a227600b2fe50b7cae11ec86d3bf1fbe471be89827e19d72aa1d"
                .to_owned(),
            private_key_wif: "5HueCGU8rMjxEXxiPuD5BDku4MkFqeZyd4dZ1jvhTVqvbTLvyTJ".to_owned(),
            address: "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa".to_owned(),
            balance_sats: 5_000_000_000,
            compressed: false,
        };
        let record = format_success_record(&hit, 123, Duration::from_secs(9));
        assert!(record.contains("total_guesses=123\n"));
        assert!(record.contains("elapsed_seconds=9\n"));
        assert!(record.contains("address=1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa\n"));
        assert!(record.contains(
            "private_key_hex=0c28fca386c7a227600b2fe50b7cae11ec86d3bf1fbe471be89827e19d72aa1d\n"
        ));
        assert!(record
            .contains("private_key_wif=5HueCGU8rMjxEXxiPuD5BDku4MkFqeZyd4dZ1jvhTVqvbTLvyTJ\n"));
    }

    #[test]
    fn toy_hash_is_stable() {
        assert_eq!(toy_hash_u64(0), 0xe220_a839_7b1d_cdaf);
        assert_eq!(toy_hash_u64(100_000), 0x5629_9769_b887_b354);
    }

    fn from_hex_32(input: &str) -> [u8; 32] {
        assert_eq!(input.len(), 64);
        let mut out = [0_u8; 32];
        for i in 0..32 {
            out[i] = u8::from_str_radix(&input[i * 2..i * 2 + 2], 16).unwrap();
        }
        out
    }
}
