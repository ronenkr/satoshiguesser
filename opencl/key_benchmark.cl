static ulong mix64(ulong x) {
    x += 0x9e3779b97f4a7c15UL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9UL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebUL;
    return x ^ (x >> 31);
}

static void copy8(uint r[8], const uint a[8]) {
    for (int i = 0; i < 8; i++) r[i] = a[i];
}

static void zero8(uint r[8]) {
    for (int i = 0; i < 8; i++) r[i] = 0u;
}

static int is_zero8(const uint a[8]) {
    uint v = 0u;
    for (int i = 0; i < 8; i++) v |= a[i];
    return v == 0u;
}

static int cmp8(const uint a[8], const uint b[8]) {
    for (int i = 7; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

static void raw_sub8(uint r[8], const uint a[8], const uint b[8]) {
    ulong borrow = 0UL;
    for (int i = 0; i < 8; i++) {
        ulong bi = (ulong)b[i] + borrow;
        ulong ai = (ulong)a[i];
        r[i] = (uint)(ai - bi);
        borrow = ai < bi;
    }
}

static void load_p(uint p[8]) {
    p[0] = 0xfffffc2fu; p[1] = 0xfffffffeu; p[2] = 0xffffffffu; p[3] = 0xffffffffu;
    p[4] = 0xffffffffu; p[5] = 0xffffffffu; p[6] = 0xffffffffu; p[7] = 0xffffffffu;
}

static void load_gx(uint gx[8]) {
    gx[0] = 0x16f81798u; gx[1] = 0x59f2815bu; gx[2] = 0x2dce28d9u; gx[3] = 0x029bfcdbu;
    gx[4] = 0xce870b07u; gx[5] = 0x55a06295u; gx[6] = 0xf9dcbbacu; gx[7] = 0x79be667eu;
}

static void load_gy(uint gy[8]) {
    gy[0] = 0xfb10d4b8u; gy[1] = 0x9c47d08fu; gy[2] = 0xa6855419u; gy[3] = 0xfd17b448u;
    gy[4] = 0x0e1108a8u; gy[5] = 0x5da4fbfcu; gy[6] = 0x26a3c465u; gy[7] = 0x483ada77u;
}

static void normalize12(ulong a[12]) {
    for (int i = 0; i < 11; i++) {
        ulong carry = a[i] >> 32;
        a[i] &= 0xffffffffUL;
        a[i + 1] += carry;
    }
}

static void reduce512(uint r[8], const uint t[16]) {
    ulong a[12];
    for (int i = 0; i < 12; i++) a[i] = 0UL;
    for (int i = 0; i < 8; i++) a[i] = (ulong)t[i];

    for (int i = 0; i < 8; i++) {
        ulong hi = (ulong)t[8 + i];
        a[i] += hi * 977UL;
        a[i + 1] += hi;
    }
    normalize12(a);

    for (int pass = 0; pass < 4; pass++) {
        for (int k = 11; k >= 8; k--) {
            ulong hi = a[k];
            a[k] = 0UL;
            if (hi != 0UL) {
                a[k - 8] += hi * 977UL;
                a[k - 7] += hi;
            }
        }
        normalize12(a);
    }

    for (int i = 0; i < 8; i++) r[i] = (uint)a[i];
    uint p[8];
    load_p(p);
    while (cmp8(r, p) >= 0) {
        uint tmp[8];
        raw_sub8(tmp, r, p);
        copy8(r, tmp);
    }
}

static void add_mod(uint r[8], const uint a[8], const uint b[8]) {
    uint t[16];
    for (int i = 0; i < 16; i++) t[i] = 0u;
    ulong carry = 0UL;
    for (int i = 0; i < 8; i++) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry;
        t[i] = (uint)sum;
        carry = sum >> 32;
    }
    t[8] = (uint)carry;
    reduce512(r, t);
}

static void sub_mod(uint r[8], const uint a[8], const uint b[8]) {
    if (cmp8(a, b) >= 0) {
        raw_sub8(r, a, b);
    } else {
        uint p[8];
        uint d[8];
        load_p(p);
        raw_sub8(d, b, a);
        raw_sub8(r, p, d);
    }
}

static void mul_mod(uint r[8], const uint a[8], const uint b[8]) {
    uint t[16];
    for (int i = 0; i < 16; i++) t[i] = 0u;

    for (int i = 0; i < 8; i++) {
        ulong carry = 0UL;
        for (int j = 0; j < 8; j++) {
            ulong cur = (ulong)t[i + j] + ((ulong)a[i] * (ulong)b[j]) + carry;
            t[i + j] = (uint)cur;
            carry = cur >> 32;
        }
        int k = i + 8;
        while (carry != 0UL && k < 16) {
            ulong cur = (ulong)t[k] + carry;
            t[k] = (uint)cur;
            carry = cur >> 32;
            k++;
        }
    }

    reduce512(r, t);
}

static void sqr_mod(uint r[8], const uint a[8]) {
    mul_mod(r, a, a);
}

static int exp_p_minus_2_bit(int bit) {
    const uint E[8] = {
        0xfffffc2du, 0xfffffffeu, 0xffffffffu, 0xffffffffu,
        0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu
    };
    return (int)((E[bit >> 5] >> (bit & 31)) & 1u);
}

static void inv_mod(uint r[8], const uint a[8]) {
    uint result[8];
    uint base[8];
    zero8(result);
    result[0] = 1u;
    copy8(base, a);

    for (int bit = 255; bit >= 0; bit--) {
        uint tmp[8];
        sqr_mod(tmp, result);
        copy8(result, tmp);
        if (exp_p_minus_2_bit(bit)) {
            mul_mod(tmp, result, base);
            copy8(result, tmp);
        }
    }
    copy8(r, result);
}

static void point_double(uint X[8], uint Y[8], uint Z[8], int *inf) {
    if (*inf || is_zero8(Y)) {
        *inf = 1;
        return;
    }

    uint A[8], B[8], C[8], D[8], E[8], F[8], T[8], T2[8];
    sqr_mod(A, X);
    sqr_mod(B, Y);
    sqr_mod(C, B);

    add_mod(T, X, B);
    sqr_mod(T, T);
    sub_mod(T, T, A);
    sub_mod(T, T, C);
    add_mod(D, T, T);

    add_mod(E, A, A);
    add_mod(E, E, A);
    sqr_mod(F, E);

    add_mod(T, D, D);
    sub_mod(T, F, T);

    sub_mod(T2, D, T);
    mul_mod(T2, E, T2);
    for (int i = 0; i < 3; i++) add_mod(C, C, C);
    sub_mod(T2, T2, C);

    mul_mod(Z, Y, Z);
    add_mod(Z, Z, Z);
    copy8(X, T);
    copy8(Y, T2);
}

static void point_add_mixed(uint X[8], uint Y[8], uint Z[8], int *inf, const uint QX[8], const uint QY[8]) {
    if (*inf) {
        copy8(X, QX);
        copy8(Y, QY);
        zero8(Z);
        Z[0] = 1u;
        *inf = 0;
        return;
    }

    uint Z2[8], U2[8], S2[8], H[8], HH[8], I[8], J[8], R[8], V[8];
    uint T[8], T2[8], X3[8], Y3[8], Z3[8];

    sqr_mod(Z2, Z);
    mul_mod(U2, QX, Z2);
    mul_mod(T, Z2, Z);
    mul_mod(S2, QY, T);
    sub_mod(H, U2, X);
    sub_mod(R, S2, Y);

    if (is_zero8(H)) {
        if (is_zero8(R)) {
            point_double(X, Y, Z, inf);
        } else {
            *inf = 1;
        }
        return;
    }

    add_mod(R, R, R);
    sqr_mod(HH, H);
    add_mod(I, HH, HH);
    add_mod(I, I, I);
    mul_mod(J, H, I);
    mul_mod(V, X, I);

    sqr_mod(X3, R);
    sub_mod(X3, X3, J);
    sub_mod(X3, X3, V);
    sub_mod(X3, X3, V);

    sub_mod(Y3, V, X3);
    mul_mod(Y3, R, Y3);
    mul_mod(T, Y, J);
    add_mod(T, T, T);
    sub_mod(Y3, Y3, T);

    add_mod(T, Z, H);
    sqr_mod(T, T);
    sub_mod(T, T, Z2);
    sub_mod(Z3, T, HH);

    copy8(X, X3);
    copy8(Y, Y3);
    copy8(Z, Z3);
}

static int scalar_bit(const uint k[8], int bit) {
    return (int)((k[bit >> 5] >> (bit & 31)) & 1u);
}

static void scalar_from_index(uint k[8], ulong index) {
    for (int i = 0; i < 4; i++) {
        ulong v = mix64(index + (ulong)i * 0x9e3779b97f4a7c15UL);
        k[i * 2] = (uint)v;
        k[i * 2 + 1] = (uint)(v >> 32);
    }
    k[7] &= 0x7fffffffu;
    if (is_zero8(k)) k[0] = 1u;
}

static void scalar_mul_g(uint AX[8], uint AY[8], const uint k[8]) {
    uint X[8], Y[8], Z[8], gx[8], gy[8];
    load_gx(gx);
    load_gy(gy);
    zero8(X);
    zero8(Y);
    zero8(Z);
    int inf = 1;

    for (int bit = 255; bit >= 0; bit--) {
        if (!inf) point_double(X, Y, Z, &inf);
        if (scalar_bit(k, bit)) point_add_mixed(X, Y, Z, &inf, gx, gy);
    }

    uint zi[8], zi2[8], zi3[8];
    inv_mod(zi, Z);
    sqr_mod(zi2, zi);
    mul_mod(zi3, zi2, zi);
    mul_mod(AX, X, zi2);
    mul_mod(AY, Y, zi3);
}

#define ROTR32(x,n) rotate((x), (uint)(32 - (n)))
#define SHR(x,n) ((x) >> (n))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define BSIG0(x) (ROTR32((x),2) ^ ROTR32((x),13) ^ ROTR32((x),22))
#define BSIG1(x) (ROTR32((x),6) ^ ROTR32((x),11) ^ ROTR32((x),25))
#define SSIG0(x) (ROTR32((x),7) ^ ROTR32((x),18) ^ SHR((x),3))
#define SSIG1(x) (ROTR32((x),17) ^ ROTR32((x),19) ^ SHR((x),10))

static uint sha_k(int i) {
    const uint K[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    return K[i];
}

static uchar pubkey_byte(const uint x[8], const uint y[8], int pos) {
    if (pos == 0) return (uchar)(2u | (y[0] & 1u));
    int limb = 7 - ((pos - 1) >> 2);
    int shift = 24 - (((pos - 1) & 3) * 8);
    return (uchar)((x[limb] >> shift) & 0xffu);
}

static void sha256_pubkey(const uint x[8], const uint y[8], uint out[8]) {
    uint w[64];
    for (int i = 0; i < 64; i++) w[i] = 0u;

    for (int i = 0; i < 33; i++) {
        int word = i >> 2;
        int shift = 24 - ((i & 3) * 8);
        w[word] |= ((uint)pubkey_byte(x, y, i)) << shift;
    }
    w[8] |= 0x00800000u;
    w[15] = 264u;

    for (int i = 16; i < 64; i++) {
        w[i] = SSIG1(w[i - 2]) + w[i - 7] + SSIG0(w[i - 15]) + w[i - 16];
    }

    uint a = 0x6a09e667u, b = 0xbb67ae85u, c = 0x3c6ef372u, d = 0xa54ff53au;
    uint e = 0x510e527fu, f = 0x9b05688cu, g = 0x1f83d9abu, h = 0x5be0cd19u;
    for (int i = 0; i < 64; i++) {
        uint t1 = h + BSIG1(e) + CH(e, f, g) + sha_k(i) + w[i];
        uint t2 = BSIG0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    out[0] = a + 0x6a09e667u;
    out[1] = b + 0xbb67ae85u;
    out[2] = c + 0x3c6ef372u;
    out[3] = d + 0xa54ff53au;
    out[4] = e + 0x510e527fu;
    out[5] = f + 0x9b05688cu;
    out[6] = g + 0x1f83d9abu;
    out[7] = h + 0x5be0cd19u;
}

#define ROL32(x,n) rotate((x), (uint)(n))

static uint ripemd_f(int j, uint x, uint y, uint z) {
    if (j < 16) return x ^ y ^ z;
    if (j < 32) return (x & y) | (~x & z);
    if (j < 48) return (x | ~y) ^ z;
    if (j < 64) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}

static uint ripemd_kl(int j) {
    if (j < 16) return 0x00000000u;
    if (j < 32) return 0x5a827999u;
    if (j < 48) return 0x6ed9eba1u;
    if (j < 64) return 0x8f1bbcdcu;
    return 0xa953fd4eu;
}

static uint ripemd_kr(int j) {
    if (j < 16) return 0x50a28be6u;
    if (j < 32) return 0x5c4dd124u;
    if (j < 48) return 0x6d703ef3u;
    if (j < 64) return 0x7a6d76e9u;
    return 0x00000000u;
}

static uchar sha_byte(const uint sha[8], int pos) {
    int word = pos >> 2;
    int shift = 24 - ((pos & 3) * 8);
    return (uchar)((sha[word] >> shift) & 0xffu);
}

static void ripemd160_sha(const uint sha[8], uint out[5]) {
    const uchar RL[80] = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
        3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
        4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
    };
    const uchar RR[80] = {
        5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
        15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
        12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
    };
    const uchar SL[80] = {
        11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
        11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
        9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
    };
    const uchar SR[80] = {
        8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
        9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
        8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
    };

    uint X[16];
    for (int i = 0; i < 16; i++) X[i] = 0u;
    for (int i = 0; i < 32; i++) X[i >> 2] |= ((uint)sha_byte(sha, i)) << ((i & 3) * 8);
    X[8] |= 0x00000080u;
    X[14] = 256u;

    uint h0 = 0x67452301u, h1 = 0xefcdab89u, h2 = 0x98badcfeu, h3 = 0x10325476u, h4 = 0xc3d2e1f0u;
    uint al = h0, bl = h1, cl = h2, dl = h3, el = h4;
    uint ar = h0, br = h1, cr = h2, dr = h3, er = h4;

    for (int j = 0; j < 80; j++) {
        uint tl = ROL32(al + ripemd_f(j, bl, cl, dl) + X[RL[j]] + ripemd_kl(j), SL[j]) + el;
        al = el; el = dl; dl = ROL32(cl, 10); cl = bl; bl = tl;
        uint tr = ROL32(ar + ripemd_f(79 - j, br, cr, dr) + X[RR[j]] + ripemd_kr(j), SR[j]) + er;
        ar = er; er = dr; dr = ROL32(cr, 10); cr = br; br = tr;
    }

    uint t = h1 + cl + dr;
    h1 = h2 + dl + er;
    h2 = h3 + el + ar;
    h3 = h4 + al + br;
    h4 = h0 + bl + cr;
    h0 = t;

    out[0] = h0; out[1] = h1; out[2] = h2; out[3] = h3; out[4] = h4;
}

__kernel void key_benchmark(
    ulong base,
    ulong stride,
    uint target0,
    uint target1,
    uint target2,
    uint target3,
    uint target4,
    uint iterations_per_item,
    __global ulong* checksum_out
) {
    ulong gid = (ulong)get_global_id(0);
    ulong work_size = (ulong)get_global_size(0);
    ulong acc = base ^ stride ^ gid;

    for (uint i = 0; i < iterations_per_item; i++) {
        ulong index = base + (gid + ((ulong)i * work_size)) * stride;
        uint k[8], x[8], y[8], sha[8], h160[5];
        scalar_from_index(k, index);
        scalar_mul_g(x, y, k);
        sha256_pubkey(x, y, sha);
        ripemd160_sha(sha, h160);

        uint matched = (uint)(
            h160[0] == target0 &&
            h160[1] == target1 &&
            h160[2] == target2 &&
            h160[3] == target3 &&
            h160[4] == target4
        );
        acc ^= ((ulong)h160[0] << 32) ^ (ulong)h160[1] ^ ((ulong)matched << 63);
    }

    checksum_out[gid] = acc;
}
