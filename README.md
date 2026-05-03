# Satoshi Guesser

A fast Rust CLI that continuously guesses random Bitcoin private keys and
checks both P2PKH address forms against the Satoshi/Patoshi address set in
`data/wallets.csv`.

The odds are still astronomically remote. This is a compute experiment, not a
practical way to recover coins.

## Build

Requires a recent stable Rust toolchain.

```bash
cargo build --release
```

Release builds use target-specific CPU tuning from `.cargo/config.toml`. x86_64
Linux, Windows, and Intel macOS builds use `target-cpu=native`; Apple Silicon
builds use an Apple ARM profile with NEON, AES, SHA2/SHA3, CRC, LSE, dot-product,
and FP16 enabled. On non-MSVC targets, the `sha2` assembly backend is also
enabled for SHA acceleration, including its Apple AArch64 SHA-256 assembly path;
MSVC builds use RustCrypto's runtime-detected SHA intrinsics instead. The wallet
CSV is embedded into the binary at compile time by default, so the release
executable does not need extra data files.

### Apple Silicon

On an M-series Mac, build normally:

```bash
cargo build --release
```

The `.cargo/config.toml` file enables an Apple Silicon CPU profile plus ARM
SIMD/crypto features. The `sha2` backend uses its Apple AArch64 assembly
implementation. At startup, Apple Silicon builds report detected SIMD/crypto
features, for example:

```text
cpu_features=apple-silicon,neon,aes,sha2,sha3,crc,lse,dotprod,fp16
```

If you are cross-compiling a portable macOS binary from another machine, remove
or override the `aarch64-apple-darwin` section in `.cargo/config.toml`.

For maximum tuning on the exact M-series machine you are building on, you can
temporarily change the Apple target CPU in `.cargo/config.toml` from `apple-m1`
to `native`.

## Run

```bash
./target/release/satoshi-guesser
```

On Windows:

```powershell
.\target\release\satoshi-guesser.exe
```

By default the program uses all available CPU cores and prints one stats line
every 5 seconds:

```text
stats elapsed=5s total_guesses=1234567 guesses_per_second=246913 average_guesses_per_second=246913
```

At startup it also prints detected CPU features, for example:

```text
loaded 21954 targets (... BTC); starting 32 worker threads; cpu_features=sse2,ssse3,sse4.1,avx2,sha-ni
```

On Apple Silicon this line reports ARM features such as `neon`, `aes`, `sha2`,
`sha3`, `crc`, `lse`, `dotprod`, and `fp16`.

Useful options:

```text
--threads N             number of worker threads, default: all logical cores
--stats-seconds N       stats interval, default: 5
--targets wallets.csv   load a CSV at runtime instead of the embedded file
--success-file path     where to write a hit, default: satoshi-guesser-success.txt
--compressed-only       only check compressed public-key addresses
--uncompressed-only     only check uncompressed public-key addresses
--toy-gpu-demo          run a synthetic u64 CUDA/OpenCL/CPU search demo instead
```

For a dedicated server, pin the thread count to the number of cores you want to
burn:

```bash
./target/release/satoshi-guesser --threads 32
```

## What It Does

Each worker thread:

1. Seeds a per-thread ChaCha20 CSPRNG from the OS RNG.
2. Generates a random 256-bit candidate private key.
3. Rejects the tiny fraction of candidates outside the secp256k1 private-key
   range.
4. Derives the secp256k1 public key.
5. Computes HASH160 for compressed and uncompressed public keys.
6. Binary-searches the embedded sorted target set.

The global guess counter is batched to reduce atomic contention. The hot path
does not allocate.

If a hit ever occurs, the process stops all workers, prints the private key hex
and WIF, matched address, balance, address form, and total guess count, then
writes the same result to `satoshi-guesser-success.txt` unless `--success-file`
points somewhere else.

## Synthetic GPU Demo

The real Bitcoin-address search path is CPU-only. For testing GPU scheduling,
there is a separate synthetic demo that searches an internally generated `u64`
nonce space. It does not use `data/wallets.csv`, Bitcoin addresses, secp256k1,
or private keys.

### CUDA

On Ubuntu with the NVIDIA driver and CUDA toolkit installed:

```bash
cargo build --release --features cuda-toy
```

Tesla P40 is compute capability 6.1, so the default CUDA architecture is
`sm_61`. To override it:

```bash
CUDA_ARCH=sm_61 cargo build --release --features cuda-toy
```

Run CPU workers plus all detected CUDA devices:

```bash
./target/release/satoshi-guesser --toy-gpu-demo --threads "$(nproc)"
```

For a quick smoke test:

```bash
./target/release/satoshi-guesser --toy-gpu-demo --threads 4 --toy-target-nonce 100000000
```

Toy CUDA tuning flags:

```text
--toy-target-nonce N    synthetic nonce to find, default: 500000000
--toy-cuda-blocks N     CUDA blocks per launch, default: 1024
--toy-cuda-threads N    CUDA threads per block, default: 256
--toy-cuda-iters N      loop iterations per CUDA thread, default: 256
```

If the binary was not compiled with `--features cuda-toy`, the toy mode still
runs on CPU and prints a message explaining how to enable CUDA.

CUDA is NVIDIA-only and is not available on Apple Silicon. The synthetic toy
mode still runs on Apple CPUs and benefits from native ARM code generation.

### OpenCL

The OpenCL toy backend is for general GPU compute devices such as Intel iGPUs.
It is not Intel Quick Sync; Quick Sync is a video encode/decode block and is not
useful for this kind of synthetic search. On Ubuntu, install an OpenCL ICD and
headers. Package names vary by Ubuntu and Intel GPU generation, but common
starting points are:

```bash
sudo apt install ocl-icd-opencl-dev clinfo intel-opencl-icd
clinfo
```

HD 610-class iGPUs usually use Intel's modern OpenCL runtime. Older Broadwell
parts such as HD 5300 may need an older or distro-specific Intel OpenCL package.
`clinfo` should list the GPU before the Rust binary can use it.

Build with OpenCL toy support:

```bash
cargo build --release --features opencl-toy
```

Build with both CUDA and OpenCL toy support:

```bash
cargo build --release --features "cuda-toy opencl-toy"
```

Run CPU workers plus all detected CUDA and OpenCL GPU devices:

```bash
./target/release/satoshi-guesser --toy-gpu-demo --threads "$(nproc)"
```

Toy OpenCL tuning flags:

```text
--toy-opencl-global N   OpenCL global work-items per launch, default: 262144
--toy-opencl-local N    OpenCL local work-items per group, default: 256
--toy-opencl-iters N    loop iterations per OpenCL work-item, default: 256
```

The global work-item count must be a multiple of the local work-item count. If
the binary was not compiled with `--features opencl-toy`, toy mode still runs on
CPU/CUDA and prints a message explaining how to enable OpenCL.

## Data

`data/wallets.csv` contains the Patoshi-pattern coinbase outputs plus the
genesis address. The expected CSV format is:

```csv
address,balance_btc
1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa,50.00000000
```

Only mainnet P2PKH addresses are loaded.

## Tests

```bash
cargo test
```

The tests cover Base58Check address decoding, BTC-to-sats parsing, the embedded
target set, and a known private-key-to-HASH160 vector.

## Performance Notes

Use `--release`. Debug builds are intentionally much slower.

The bottleneck is secp256k1 public-key derivation, so throughput scales mostly
with CPU cores. SHA-256 uses the assembly backend when the CPU supports it; the
target lookup is a sorted in-memory binary search over about 22k HASH160 values
and is tiny compared with elliptic-curve work.

If you need a portable binary for mixed CPU generations, remove
`.cargo/config.toml` or replace `target-cpu=native` with a baseline CPU before
building.
