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

Release builds use `target-cpu=native`, so build on the server that will run the
binary. That lets LLVM emit instructions for available CPU features such as
AVX2, SSE4.1, SHA-NI, and their ARM equivalents. On non-MSVC targets, the
`sha2` assembly backend is also enabled for SHA acceleration; MSVC builds use
RustCrypto's runtime-detected SHA intrinsics instead. The wallet CSV is embedded
into the binary at compile time by default, so the release executable does not
need extra data files.

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

Useful options:

```text
--threads N             number of worker threads, default: all logical cores
--stats-seconds N       stats interval, default: 5
--targets wallets.csv   load a CSV at runtime instead of the embedded file
--success-file path     where to write a hit, default: satoshi-guesser-success.txt
--compressed-only       only check compressed public-key addresses
--uncompressed-only     only check uncompressed public-key addresses
--toy-gpu-demo          run a synthetic u64 CUDA/CPU search demo instead
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

## Synthetic CUDA Demo

The real Bitcoin-address search path is CPU-only. For testing your Tesla P40s,
there is a separate synthetic demo that searches an internally generated `u64`
nonce space. It does not use `data/wallets.csv`, Bitcoin addresses, secp256k1,
or private keys.

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
