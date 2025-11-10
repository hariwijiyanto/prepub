import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time
import os
import sys
import argparse
import struct
import random
from ecdsa import SigningKey, SECP256k1
from ecdsa.ellipticcurve import Point, CurveFp

def int_to_bigint_np(val):
    bigint_arr = np.zeros(8, dtype=np.uint32)
    for j in range(8):
        bigint_arr[j] = (val >> (32 * j)) & 0xFFFFFFFF
    return bigint_arr

def bigint_np_to_int(bigint_arr):
    val = 0
    for j in range(8):
        val |= int(bigint_arr[j]) << (32 * j)
    return val

def splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return (x ^ (x >> 31)) & ((1 << 64) - 1)

def make_fp(x, y_parity):
    low64 = x & ((1 << 64) - 1)
    return splitmix64(low64 ^ y_parity)

def load_target_pubkeys(filename):
    targets_bin = bytearray()
    is_text_file = filename.lower().endswith('.txt')

    with open(filename, 'r' if is_text_file else 'rb') as f:
        if is_text_file:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if len(line) == 66 and line[:2] in ('02', '03'):
                    try:
                        pubkey_bytes = bytes.fromhex(line)
                        targets_bin.extend(pubkey_bytes)
                    except ValueError:
                        print(f"[!] Warning: Invalid hex format on line {line_num}: {line}")
                elif line and not line.startswith('#'):
                    print(f"[!] Warning: Invalid format on line {line_num}: {line}")
        else:
            while True:
                chunk = f.read(33)
                if not chunk:
                    break
                if len(chunk) == 33:
                    targets_bin.extend(chunk)

    return targets_bin

def init_secp256k1_constants(mod):
    # Prime modulus p
    p_data = np.array([
        0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    ], dtype=np.uint32)
    const_p_gpu = mod.get_global("const_p")[0]
    cuda.memcpy_htod(const_p_gpu, p_data)

    # Curve order n
    n_data = np.array([
        0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
        0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    ], dtype=np.uint32)
    const_n_gpu = mod.get_global("const_n")[0]
    cuda.memcpy_htod(const_n_gpu, n_data)

    # Base point G in Jacobian coordinates
    g_x = np.array([
        0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
        0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
    ], dtype=np.uint32)
    g_y = np.array([
        0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
        0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
    ], dtype=np.uint32)
    g_z = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
    g_infinity = np.array([False], dtype=np.bool_)

    # Create structured array for ECPointJac
    ecpoint_jac_dtype = np.dtype([
        ('X', np.uint32, 8),
        ('Y', np.uint32, 8),
        ('Z', np.uint32, 8),
        ('infinity', np.bool_)
    ])
    g_jac = np.zeros(1, dtype=ecpoint_jac_dtype)
    g_jac['X'], g_jac['Y'], g_jac['Z'], g_jac['infinity'] = g_x, g_y, g_z, g_infinity

    const_G_gpu = mod.get_global("const_G_jacobian")[0]
    cuda.memcpy_htod(const_G_gpu, g_jac)

def run_precomputation(mod):
    precompute_kernel = mod.get_function("precompute_G_table_kernel")
    precompute_kernel(block=(1, 1, 1))
    cuda.Context.synchronize()
    print("[*] Precomputation table selesai.")

def decompress_pubkey(compressed):
    if len(compressed) != 33:
        raise ValueError("Compressed public key should be 33 bytes")

    prefix = compressed[0]
    x_bytes = compressed[1:]

    if prefix not in [0x02, 0x03]:
        raise ValueError("Invalid prefix for compressed public key")

    # Convert x bytes to integer
    x_int = int.from_bytes(x_bytes, 'big')

    # Calculate y^2 = x^3 + 7 mod p
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    a = 0
    b = 7

    x3 = pow(x_int, 3, p)
    y_sq = (x3 + a * x_int + b) % p

    # Calculate square root of y_sq mod p
    y = pow(y_sq, (p + 1) // 4, p)

    # Check parity
    is_even = (y % 2 == 0)
    if (prefix == 0x02 and not is_even) or (prefix == 0x03 and is_even):
        y = p - y

    return x_int, y

def optimize_bloom_filter_size(num_items, false_positive_rate=0.001):
    # Calculate optimal bloom filter size
    m = - (num_items * np.log(false_positive_rate)) / (np.log(2) ** 2)
    # Round up to nearest power of two for better GPU performance
    return int(2 ** np.ceil(np.log2(m)))

def calculate_dp_point(target_x, target_y, reduction_factor):
    """
    Menghitung DP point dengan operasi kurva elliptic yang benar:
    DP = Target - reduction_factor * G
    """
    # Gunakan library ecdsa untuk perhitungan yang benar
    curve = SECP256k1.curve
    G = SECP256k1.generator
    target_point = Point(curve, target_x, target_y)

    # Hitung reduction_factor * G
    reduction_g_point = reduction_factor * G

    # Hitung Target - reduction_factor * G
    negative_reduction_g = Point(curve, reduction_g_point.x(), -reduction_g_point.y() % curve.p())
    dp_point = target_point + negative_reduction_g

    return dp_point.x(), dp_point.y()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CUDA Compressed Public Key Brute Forcer with Optimized DP')
    parser.add_argument('--start', type=lambda x: int(x, 0), required=True, help='Kunci privat awal')
    parser.add_argument('--end', type=lambda x: int(x, 0), required=True, help='Kunci privat akhir')
    parser.add_argument('--step', type=lambda x: int(x, 0), default=1, help='Langkah penambahan')
    parser.add_argument('--file', required=True, help='File target public keys (format: .txt untuk hex, .bin untuk binary)')
    parser.add_argument('--keys-per-launch', type=int, default=2**20, help='Jumlah kunci per batch GPU (default: 1,048,576)')
    parser.add_argument('--reduction-step', type=int, default=1000000, help='Langkah pengurangan untuk membuat DP')
    parser.add_argument('--num-dp', type=int, default=1000000, help='Jumlah DP points (default: 100,000)')  # Increased number of DP
    parser.add_argument('--bloom-fp-rate', type=float, default=0.001, help='False positive rate untuk Bloom Filter (default: 0.001)')  # Lower FP rate
    parser.add_argument('--no-dp', action='store_true', help='Nonaktifkan DP untuk benchmarking')
    args = parser.parse_args()

    # Inisialisasi
    print("[*] Memuat file kernel CUDA...")
    try:
        with open('kernel.cu', 'r') as f:
            full_cuda_code = f.read()
    except FileNotFoundError:
        print("[!] FATAL: File 'kernel_pubkey_optimized.cu' tidak ditemukan.")
        sys.exit(1)

    print("[*] Mengkompilasi kernel CUDA...")
    mod = SourceModule(full_cuda_code, no_extern_c=False, options=['-std=c++11', '-arch=sm_75'])
    init_secp256k1_constants(mod)

    # Jalankan precomputation
    print("[*] Menjalankan precomputation...")
    run_precomputation(mod)

    # Dapatkan kernel function berdasarkan mode
    if args.no_dp:
        find_pubkey_kernel = mod.get_function("find_pubkey_kernel_no_dp")
        print("[*] Mode tanpa DP diaktifkan")
    else:
        find_pubkey_kernel = mod.get_function("find_pubkey_kernel_with_dp")
        print("[*] Mode dengan DP diaktifkan")

    print("[*] Inisialisasi selesai.")

    # Muat target public keys
    target_bin = load_target_pubkeys(args.file)
    num_targets = len(target_bin) // 33
    if num_targets == 0:
        print(f"[!] Tidak ada public key valid di {args.file}.")
        sys.exit(1)

    print(f"[*] Berhasil memuat {num_targets} public key target.")

    # Decompress target public key
    target_compressed = target_bin[:33]
    target_x, target_y = decompress_pubkey(target_compressed)
    print(f"[*] Target public key: {target_compressed.hex()}")
    print(f"[*] Target coordinates: x={hex(target_x)}, y={hex(target_y)}")

    # Inisialisasi struktur data
    d_target_pubkeys = cuda.mem_alloc(len(target_bin))
    cuda.memcpy_htod(d_target_pubkeys, np.frombuffer(target_bin, dtype=np.uint8))

    d_result = cuda.mem_alloc(32)
    d_found_flag = cuda.mem_alloc(4)
    cuda.memset_d32(d_result, 0, 8)
    cuda.memset_d32(d_found_flag, 0, 1)

    # Inisialisasi DP dan Bloom Filter hanya jika tidak dinonaktifkan
    d_bloom_filter = None
    d_dp_table = None
    bloom_size = 0
    dp_table_size = 0

    if not args.no_dp:
        # Hitung ukuran Bloom Filter yang optimal
        bloom_size = optimize_bloom_filter_size(args.num_dp, args.bloom_fp_rate)
        print(f"[*] Menggunakan Bloom Filter dengan ukuran {bloom_size} bits untuk {args.num_dp} DP (FP rate: {args.bloom_fp_rate})")

        # Inisialisasi Bloom Filter
        bloom_filter = np.zeros((bloom_size + 31) // 32, dtype=np.uint32)

        # Inisialisasi DP Table
        dp_table_dtype = np.dtype([
            ('fp', np.uint64),
            ('reduction_factor', np.uint64)
        ])
        dp_table = np.zeros(args.num_dp, dtype=dp_table_dtype)

        # Precompute DP points dengan perhitungan yang benar
        print(f"[*] Precomputing {args.num_dp} DP points dengan reduction step {args.reduction_step}...")

        start_time_dp = time.time()
        last_print_time = start_time_dp
        print_interval = 1  # print setiap 1 detik

        for i in range(1, args.num_dp + 1):
            reduction_factor = i * args.reduction_step
            if reduction_factor > 2**64 - 1:
                reduction_factor = 2**64 - 1

            # Hitung titik P_i = target - i * reduction_step * G dengan operasi kurva yang benar
            try:
                dp_x, dp_y = calculate_dp_point(target_x, target_y, reduction_factor)
                dp_y_parity = 1 if dp_y % 2 else 0

                fp = make_fp(dp_x, dp_y_parity)

                # Update Bloom Filter
                h1 = splitmix64(fp)
                h2 = splitmix64(h1)
                for j in range(4):
                    index = (h1 + j * h2) % bloom_size
                    word_idx = index // 32
                    bit_idx = index % 32
                    bloom_filter[word_idx] |= (1 << bit_idx)

                # Update DP Table
                dp_table[i-1]['fp'] = fp
                dp_table[i-1]['reduction_factor'] = reduction_factor
            except Exception as e:
                print(f"[!] Error calculating DP point {i}: {e}")
                continue

            # Progress reporting dengan estimasi waktu
            current_time = time.time()
            if current_time - last_print_time >= print_interval or i == args.num_dp:
                progress = i / args.num_dp * 100
                elapsed = current_time - start_time_dp
                eta = (elapsed / i) * (args.num_dp - i) if i > 0 else 0

                sys.stdout.write(f"\r[*] Generating DP: {progress:.1f}% | "
                                f"ETA: {eta:.1f}s | "
                                f"Elapsed: {elapsed:.1f}s")
                sys.stdout.flush()
                last_print_time = current_time

        sys.stdout.write("\n")
        print(f"[*] DP generation completed in {time.time() - start_time_dp:.1f} seconds")

        # Urutkan DP Table berdasarkan fp untuk binary search
        print("[*] Sorting DP table...")
        dp_table.sort(order='fp')

        # Upload ke GPU
        print("[*] Uploading Bloom filter and DP table to GPU...")
        d_bloom_filter = cuda.mem_alloc(bloom_filter.nbytes)
        cuda.memcpy_htod(d_bloom_filter, bloom_filter)

        d_dp_table = cuda.mem_alloc(dp_table.nbytes)
        cuda.memcpy_htod(d_dp_table, dp_table)
        dp_table_size = args.num_dp
        print("[*] Bloom filter and DP table uploaded successfully")

    # Inisialisasi untuk loop
    total_keys_checked = 0
    start_time = time.time()
    iteration = 0
    current_start = args.start
    current_end = args.end
    found_flag_host = np.zeros(1, dtype=np.int32)

    try:
        while found_flag_host[0] == 0 and current_end >= 0:
            print(f"\n[+] Iterasi {iteration}: Rentang {hex(current_start)} - {hex(current_end)} (Size: {(current_end - current_start)//args.step + 1:,} keys)")

            temp_current = current_start
            keys_in_window = (current_end - current_start) // args.step + 1
            keys_processed_in_window = 0

            # Loop dalam untuk memindai seluruh rentang saat ini
            while temp_current <= current_end and found_flag_host[0] == 0:
                keys_left = (current_end - temp_current) // args.step + 1
                keys_this_launch = min(args.keys_per_launch, keys_left)
                if keys_this_launch <= 0:
                    break

                start_key_np = int_to_bigint_np(temp_current)
                step_np = int_to_bigint_np(args.step)

                block_size = 256
                grid_size = (keys_this_launch + block_size - 1) // block_size

                # Luncurkan kernel berdasarkan mode
                if args.no_dp:
                    # Mode tanpa DP
                    find_pubkey_kernel(
                        cuda.In(start_key_np), np.uint64(keys_this_launch), cuda.In(step_np),
                        d_target_pubkeys, np.int32(num_targets),
                        d_result, d_found_flag,
                        block=(block_size, 1, 1), grid=(grid_size, 1)
                    )
                else:
                    # Mode dengan DP
                    find_pubkey_kernel(
                        cuda.In(start_key_np), np.uint64(keys_this_launch), cuda.In(step_np),
                        d_target_pubkeys, np.int32(num_targets),
                        d_result, d_found_flag,
                        d_bloom_filter, np.uint64(bloom_size),
                        d_dp_table, np.uint32(dp_table_size),
                        block=(block_size, 1, 1), grid=(grid_size, 1)
                    )

                cuda.Context.synchronize()

                total_keys_checked += keys_this_launch
                keys_processed_in_window += keys_this_launch

                # Periksa hasil
                cuda.memcpy_dtoh(found_flag_host, d_found_flag)

                elapsed = time.time() - start_time
                speed = total_keys_checked / elapsed if elapsed > 0 else 0
                window_progress = 100 * keys_processed_in_window / keys_in_window
                progress_str = f"[+] Total: {total_keys_checked:,} | Kecepatan: {speed:,.2f} k/s | Window: {window_progress:.1f}% | Kunci: {hex(temp_current)}"
                sys.stdout.write('\r' + progress_str.ljust(120))
                sys.stdout.flush()

                temp_current += keys_this_launch * args.step

            # Geser window ke bawah
            window_size = 33554432  # 2^25
            current_start -= window_size
            current_end -= window_size
            iteration += 1

            # Hentikan jika mencapai batas bawah
            if current_end < 0:
                print("\n[!] Batas bawah tercapai (kunci negatif)")
                break

        # Setelah loop selesai, cetak hasilnya
        if found_flag_host[0] == 1:
            sys.stdout.write('\n')
            sys.stdout.flush()
            print("\n[+] KUNCI PRIVAT DITEMUKAN!")
            found_privkey_np = np.zeros(8, dtype=np.uint32)
            cuda.memcpy_dtoh(found_privkey_np, d_result)

            privkey_int = 0
            for j in range(8):
                privkey_int |= int(found_privkey_np[j]) << (32 * j)

            print(f"    Kunci Privat: {hex(privkey_int)}")
            print(f"    Rentang ditemukan: {hex(current_start+1)} - {hex(current_end+1)} (Iterasi {iteration-1})")
            with open("found_private_key.txt", "w") as f:
                f.write(f"Private Key: {hex(privkey_int)}\n")
                f.write(f"Range: {hex(current_start+1)} - {hex(current_end+1)}\n")
        else:
            print("\n\n[+] Pencarian selesai. Tidak ada yang cocok ditemukan.")
            print(f"    Total kunci dicoba: {total_keys_checked:,}")
            print(f"    Rentang terakhir: {hex(current_start+1)} - {hex(current_end+1)}")

    except KeyboardInterrupt:
        print("\n\n[!] Dihentikan oleh pengguna")
        print(f"    Iterasi terakhir: {iteration} | Rentang: {hex(current_start)} - {hex(current_end)}")
        print(f"    Total kunci dicoba: {total_keys_checked:,}")
        sys.exit(0)
