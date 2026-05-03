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

const EMBEDDED_WALLETS: &str = include_str!("../data/wallets.csv");
const DEFAULT_STATS_SECONDS: u64 = 5;
const DEFAULT_SUCCESS_FILE: &str = "satoshi-guesser-success.txt";
const DEFAULT_TOY_TARGET_NONCE: u64 = 500_000_000;
const DEFAULT_TOY_CUDA_BLOCKS: u32 = 1024;
const DEFAULT_TOY_CUDA_THREADS: u32 = 256;
const DEFAULT_TOY_CUDA_ITERS: u32 = 256;
const SATS_PER_BTC: u64 = 100_000_000;

#[derive(Clone, Debug)]
struct Args {
    threads: usize,
    stats_seconds: u64,
    targets_path: Option<String>,
    success_file: String,
    compressed: bool,
    uncompressed: bool,
    toy_gpu_demo: bool,
    toy_target_nonce: u64,
    toy_cuda_blocks: u32,
    toy_cuda_threads: u32,
    toy_cuda_iters: u32,
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

fn main() {
    let args = parse_args().unwrap_or_else(|err| {
        eprintln!("{err}");
        eprintln!();
        print_usage_and_exit(2);
    });

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

fn run_toy_gpu_demo(args: Args) {
    let target_hash = toy_hash_u64(args.toy_target_nonce);
    let cuda_devices = toy_cuda_device_count();
    let total_lanes = args.threads + cuda_devices;

    eprintln!(
        "toy_gpu_demo target_nonce={} target_hash={:#018x}; cpu_threads={}; cuda_devices={}; cpu_features={}",
        args.toy_target_nonce,
        target_hash,
        args.threads,
        cuda_devices,
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
    total_guesses: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    hit_tx: mpsc::Sender<ToyHit>,
) {
    let mut local_count = 0_u64;
    while !stop.load(AtomicOrdering::Relaxed) {
        local_count += 1;
        let hash = toy_hash_u64(nonce);
        if hash == target_hash {
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
                report_toy_hit(&hit_tx, format!("cuda-{device}"), nonce, target_hash, &stop);
                break;
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
    let mut toy_gpu_demo = false;
    let mut toy_target_nonce = DEFAULT_TOY_TARGET_NONCE;
    let mut toy_cuda_blocks = DEFAULT_TOY_CUDA_BLOCKS;
    let mut toy_cuda_threads = DEFAULT_TOY_CUDA_THREADS;
    let mut toy_cuda_iters = DEFAULT_TOY_CUDA_ITERS;

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
            "--toy-gpu-demo" => {
                toy_gpu_demo = true;
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
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        threads,
        stats_seconds,
        targets_path,
        success_file,
        compressed,
        uncompressed,
        toy_gpu_demo,
        toy_target_nonce,
        toy_cuda_blocks,
        toy_cuda_threads,
        toy_cuda_iters,
    })
}

fn next_arg(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn print_usage_and_exit(code: i32) -> ! {
    eprintln!(
        "Usage: satoshi-guesser [--threads N] [--stats-seconds N] [--targets wallets.csv] [--success-file path] [--compressed-only|--uncompressed-only] [--toy-gpu-demo] [--toy-target-nonce N] [--toy-cuda-blocks N] [--toy-cuda-threads N] [--toy-cuda-iters N]"
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
