/* ============================================================
 * F(x)-约化下的 GF(2) 线性代数密码分析程序（优化版）
 *
 * 与原版相比的改动：
 *  [正确性] 矩阵每行改用 uint64_t 字数组位集，不再用单个
 *           uint32_t（原版 col>=32 时 1u<<col 是未定义行为，
 *           在 STATE_N=79 / N_PARAM=80 规模下结果本身是错的）。
 *  [正确性] 修正 sprime 写越界、空间问题。
 *  [性能]   L_e / L_const / 内积全部改成对 D 支撑集做位并行
 *           popcount-奇偶（__builtin_parityll），把按 D_len 的
 *           标量循环压缩约 64 倍。
 *  [性能]   预计算 eD[o]（e 在 D 偏移上的打包向量）、pe[o]
 *           （线性项奇偶）、Q[gap][o]（二次项奇偶），消除窗口
 *           间的大量重复计算。
 *  [性能]   build_M 里的 T*Br 利用 T 的稀疏（移位+反馈）结构，
 *           O(STATE_N^2) -> O(STATE_N)。
 *  [清理]   删除从不被读取的 coeff 行组合追踪（死计算）。
 * ============================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <limits.h>
#include <time.h>

/* ---- 可调参数（默认全规模，测试时可用 -D 覆盖） ---- */
#ifndef STATE_N
#define STATE_N 79
#endif
#ifndef N_PARAM
#define N_PARAM 80
#endif
#ifndef BIG_N
#define BIG_N 20000
#endif
#ifndef BIG_K
#define BIG_K 20000
#endif
#ifndef C_PARAM
#define C_PARAM 5000
#endif
#ifndef POLY_FILE
#define POLY_FILE "E:\\F2.txt"
#endif

#define MAX_D_TERMS 4096

static int D[MAX_D_TERMS];
static int D_len = 0;

#ifndef LFSR_TAPS_INIT
#define LFSR_TAPS_INIT \
    0, 1, 2, 5, 8, 10, 11, 12, 13, 14, 16, 18, 19, 22, 23, 24, 25, \
    27, 28, 31, 32, 33, 35, 36, 38, 41, 44, 46, 47, 49, 50, 51, 54, \
    59, 60, 61, 62, 63, 66, 67, 71, 72, 73, 74, 75, 76, 77, 78
#endif

static const int LFSR_TAPS[] = { LFSR_TAPS_INIT };
#define TAP_COUNT ((int)(sizeof(LFSR_TAPS) / sizeof(LFSR_TAPS[0])))
static uint8_t T_mat[STATE_N][STATE_N];
static uint8_t A_mat[STATE_N][MAX_D_TERMS];

/* 序列长度根据实际需要在运行时精确计算，避免固定 SEQ_LEN 过大。 */
static int seq_len = 0;
static int *e_seq = NULL;
static int *sprime = NULL;

/* ---- 矩阵列数与位集字宽 ---- */
#define M1_COLS ((N_PARAM + 1) + (N_PARAM * (N_PARAM - 1)) / 2)
#define M_COLS  (STATE_N * N_PARAM)
#define M1_ROWS (BIG_N + 1)
#define M_ROWS  (BIG_K + 1)

#define WORDS_M  (((M_COLS) + 63) / 64)
#define WORDS_M1 (((M1_COLS) + 63) / 64)

/* 每行一段连续字。 */
static uint64_t *M_bits;   /* M_ROWS  * WORDS_M  */
static uint64_t *M1_bits;  /* M1_ROWS * WORDS_M1 */

#define MROW(r)  (M_bits  + (size_t)(r) * WORDS_M)
#define M1ROW(r) (M1_bits + (size_t)(r) * WORDS_M1)

/* ---- D 支撑集相关的预计算缓冲 ---- */
static int      WORDS_D = 0;        /* (D_len+63)/64 */
static int      OMAX    = 0;        /* eD 索引上界 */
static uint64_t *eD     = NULL;     /* (OMAX+1) * WORDS_D */
static uint8_t  *pe     = NULL;     /* (OMAX+1)，线性项奇偶 */
static uint8_t  *Qtab   = NULL;     /* N_PARAM * (OMAX+1)，二次项奇偶 */

#define EDROW(o) (eD + (size_t)(o) * WORDS_D)
#define QAT(g, o) Qtab[(size_t)(g) * (OMAX + 1) + (o)]

/* M1 最多访问 pe[BIG_N + N_PARAM]；
 * M 最多访问 eD[BIG_K + N_PARAM - 1]。
 * 因此 OMAX 取二者最大值即可。
 */
static int required_omax(void)
{
    const int m1_max = BIG_N + N_PARAM;
    const int m_max  = BIG_K + N_PARAM - 1;
    return (m1_max > m_max) ? m1_max : m_max;
}

/* e_seq 最后会访问 e_seq[OMAX + max(D)]，因此最紧凑的长度为
 *     seq_len = OMAX + max(D) + 1
 * sprime 在生成 seq_len 项时最后写到 seq_len + N_PARAM - 2。
 */
static int allocate_sequences(void)
{
    long long needed;

    if (D_len <= 0) return 0;

    OMAX = required_omax();
    needed = (long long)OMAX + (long long)D[D_len - 1] + 1LL;
    if (needed <= 0 || needed > INT_MAX) return 0;

    seq_len = (int)needed;
    e_seq = (int *)calloc((size_t)seq_len, sizeof(int));
    sprime = (int *)calloc((size_t)seq_len + (size_t)N_PARAM - 1u, sizeof(int));

    if (!e_seq || !sprime) {
        free(e_seq);
        free(sprime);
        e_seq = NULL;
        sprime = NULL;
        seq_len = 0;
        return 0;
    }
    return 1;
}

static double wall_time_seconds(void)
{
#if defined(TIME_UTC)
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) == TIME_UTC)
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void print_step_time(const char *name, double start)
{
    printf("[TIME] %-30s : %.6f seconds\n",
           name, wall_time_seconds() - start);
}

/* ============================================================
 * 工具：位集奇偶
 * ============================================================ */

/* parity( popcount( a ) )，a 含 w 个字 */
static inline int vec_parity(const uint64_t *a, int w)
{
    int p = 0;
    for (int k = 0; k < w; k++) {
        p ^= __builtin_parityll(a[k]);
    }
    return p;
}

/* parity( sum_t a_t & b_t ) = parity( popcount(a & b) ) */
static inline int dot_parity(const uint64_t *a, const uint64_t *b, int w)
{
    int p = 0;
    for (int k = 0; k < w; k++) {
        p ^= __builtin_parityll(a[k] & b[k]);
    }
    return p;
}

/* ============================================================
 * 读取 F.txt 并提取支撑集（与原版逻辑一致）
 * ============================================================ */

static int compare_int(const void *a, const void *b)
{
    int x = *(const int *)a;
    int y = *(const int *)b;
    return (x > y) - (x < y);
}

static int read_support_set(const char *filename)
{
    FILE *fp = NULL;
    char *buffer = NULL;
    int raw_degrees[MAX_D_TERMS];
    long file_size;
    size_t bytes_read, begin = 0, write_pos = 0;
    int raw_count = 0, success = 0;

    D_len = 0;
    fp = fopen(filename, "rb");
    if (fp == NULL) goto cleanup;
    if (fseek(fp, 0, SEEK_END) != 0) goto cleanup;
    file_size = ftell(fp);
    if (file_size <= 0) goto cleanup;
    if (fseek(fp, 0, SEEK_SET) != 0) goto cleanup;

    buffer = (char *)malloc((size_t)file_size + 1u);
    if (buffer == NULL) goto cleanup;

    bytes_read = fread(buffer, 1, (size_t)file_size, fp);
    if (bytes_read != (size_t)file_size) goto cleanup;
    buffer[bytes_read] = '\0';

    if (bytes_read >= 3 &&
        (unsigned char)buffer[0] == 0xEF &&
        (unsigned char)buffer[1] == 0xBB &&
        (unsigned char)buffer[2] == 0xBF) {
        begin = 3;
    }

    for (size_t i = begin; i < bytes_read; i++) {
        unsigned char ch = (unsigned char)buffer[i];
        if (isspace(ch)) continue;
        if (ch == '-') ch = '+';
        buffer[write_pos++] = (char)ch;
    }
    buffer[write_pos] = '\0';
    if (write_pos == 0) goto cleanup;

    {
        char *term = strtok(buffer, "+");
        while (term != NULL) {
            int degree;
            if (strcmp(term, "1") == 0) {
                degree = 0;
            } else if (strcmp(term, "0") == 0) {
                term = strtok(NULL, "+");
                continue;
            } else if (strcmp(term, "x") == 0 || strcmp(term, "X") == 0) {
                degree = 1;
            } else if ((term[0] == 'x' || term[0] == 'X') && term[1] == '^') {
                char *end_ptr;
                long value;
                if (term[2] == '\0') goto cleanup;
                value = strtol(term + 2, &end_ptr, 10);
                if (*end_ptr != '\0' || value < 0 || value > INT_MAX) goto cleanup;
                degree = (int)value;
            } else {
                goto cleanup;
            }
            if (raw_count >= MAX_D_TERMS) goto cleanup;
            raw_degrees[raw_count++] = degree;
            term = strtok(NULL, "+");
        }
    }

    if (raw_count == 0) goto cleanup;
    qsort(raw_degrees, (size_t)raw_count, sizeof(raw_degrees[0]), compare_int);

    D_len = 0;
    for (int i = 0; i < raw_count;) {
        int j = i + 1;
        while (j < raw_count && raw_degrees[j] == raw_degrees[i]) j++;
        if ((j - i) & 1) D[D_len++] = raw_degrees[i];
        i = j;
    }
    if (D_len == 0) goto cleanup;
    success = 1;

cleanup:
    if (fp != NULL) fclose(fp);
    free(buffer);
    return success;
}

/* ============================================================
 * 生成状态转移矩阵 T（用于生成 A）
 * ============================================================ */

static int generate_transition_matrix(void)
{
    memset(T_mat, 0, sizeof(T_mat));
    for (int i = 1; i < STATE_N; i++) T_mat[i][i - 1] = 1;
    for (int k = 0; k < TAP_COUNT; k++) {
        int tap = LFSR_TAPS[k];
        if (tap < 0 || tap >= STATE_N) return 0;
        T_mat[tap][STATE_N - 1] = 1;
    }
    return 1;
}

static void matrix_vector_multiply_mod2(
    uint8_t matrix[STATE_N][STATE_N],
    uint8_t state[STATE_N],
    uint8_t result[STATE_N])
{
    for (int i = 0; i < STATE_N; i++) {
        uint8_t value = 0;
        for (int j = 0; j < STATE_N; j++) {
            value ^= (uint8_t)(matrix[i][j] & state[j]);
        }
        result[i] = value;
    }
}

static int generate_A_matrix(void)
{
    uint8_t state[STATE_N] = {0};
    int column = 0, max_exponent;
    if (D_len <= 0) return 0;
    memset(A_mat, 0, sizeof(A_mat));
    state[0] = 1;
    max_exponent = D[D_len - 1];

    for (int exponent = 0; exponent <= max_exponent; exponent++) {
        if (column < D_len && exponent == D[column]) {
            for (int row = 0; row < STATE_N; row++)
                A_mat[row][column] = state[row];
            column++;
            if (column == D_len) break;
        }
        {
            uint8_t next_state[STATE_N];
            matrix_vector_multiply_mod2(T_mat, state, next_state);
            memcpy(state, next_state, sizeof(state));
        }
    }
    return column == D_len;
}

/* ============================================================
 * NFSR/LFSR 序列生成（与原版完全一致）
 * ============================================================ */

static int nfsr_feedback(const int *s)
{
    int linear = s[0]^s[3]^s[4]^s[5];
    int nonlinear = s[1] & s[2];
    nonlinear = nonlinear^(s[0] & s[79]);
    nonlinear = nonlinear^(s[2] & s[3]);
    nonlinear = nonlinear^(s[3] & s[4]);
    return linear ^ nonlinear;
}

static void generate_sequences(void)
{
    int s_state[N_PARAM] = {0};
    s_state[1] = 1;
    int sp_len = N_PARAM-1;
    for(int i=0;i< STATE_N;i++)
    {
        sprime[i]=0;
    }
     sprime[1] = 1;

    for (int i = 0; i < seq_len; i++) {
        int s = nfsr_feedback(s_state);
        int s1=0;
        for (int k=0;k<TAP_COUNT;k++)
        {
            s1=s1^sprime[i+LFSR_TAPS[k]] ;
        }
      
        e_seq[i] = sprime[i] ^ s_state[0];
        for (int j = 0; j < STATE_N; j++) s_state[j] = s_state[j + 1];
        s_state[STATE_N] = s;
        sprime[sp_len++] = s1;
    }
}

/* ============================================================
 * D 偏移预计算：eD / pe / Qtab
 * ============================================================ */

static int build_D_precompute(void)
{
    OMAX = required_omax();
    WORDS_D = (D_len + 63) / 64;

    eD   = (uint64_t *)calloc((size_t)(OMAX + 1) * WORDS_D, sizeof(uint64_t));
    pe   = (uint8_t  *)calloc((size_t)(OMAX + 1), 1);
    Qtab = (uint8_t  *)calloc((size_t)N_PARAM * (OMAX + 1), 1);
    if (!eD || !pe || !Qtab) return 0;

    /* eD[o] 的第 t 位 = e_seq[o + D[t]] */
    for (int o = 0; o <= OMAX; o++) {
        uint64_t *row = EDROW(o);
        for (int t = 0; t < D_len; t++) {
            if (e_seq[o + D[t]] & 1)
                row[t >> 6] |= (uint64_t)1 << (t & 63);
        }
        pe[o] = (uint8_t)vec_parity(row, WORDS_D);   /* L_e_fn(o) */
    }

    /* Qtab[g][o] = parity( eD[o] & eD[o+g] ) = L_const_fn(o, o+g) */
    for (int g = 1; g < N_PARAM; g++) {
        for (int o = 0; o + g <= OMAX; o++) {
            QAT(g, o) = (uint8_t)dot_parity(EDROW(o), EDROW(o + g), WORDS_D);
        }
    }
    return 1;
}

/* 仅用于 main 中的 L'_{1,2}，调用次数 O(1)，保留标量实现 */
static int Lprime_fn(int i, int j)
{
    int ans = 0;
    for (int t = 0; t < D_len; t++)
        ans ^= (sprime[i + D[t]] & e_seq[j + D[t]]);
    return ans & 1;
}

static int reduced_quadratic_fn(int i, int j)
{
    /* 原版只启用 L_const_fn(i,j) */
    return dot_parity(EDROW(i), EDROW(j), WORDS_D);
}

/* ============================================================
 * 构造 M1（列数 M1_COLS，行存于 WORDS_M1 个字）
 * ============================================================ */

static void build_M1(void)
{
    for (int r = 0; r <= BIG_N; r++) {
        uint64_t *row = M1ROW(r);
        memset(row, 0, (size_t)WORDS_M1 * sizeof(uint64_t));
        int col = 0;

        /* 线性约化 */
        for (int a = 0; a <= N_PARAM; a++) {
            if (pe[r + a]) row[col >> 6] |= (uint64_t)1 << (col & 63);
            col++;
        }
        /* 二次约化：枚举顺序与原 make_quadratic_pairs 完全相同 */
        for (int gap = 1; gap < N_PARAM; gap++) {
            for (int i = 0; i < N_PARAM - gap; i++) {
                if (QAT(gap, r + i)) row[col >> 6] |= (uint64_t)1 << (col & 63);
                col++;
            }
        }
    }
}

/* ============================================================
 * 构造 M（列数 M_COLS = STATE_N*N_PARAM）
 *
 * Br = T^r A，按行打包到 WORDS_D 个字；
 * value(i) = parity( Br[i] & eD[r+b] )；
 * T*Br 利用稀疏结构（下移一行 + 反馈到 tap 行）。
 * ============================================================ */

static void build_M(void)
{
    uint64_t *Br   = (uint64_t *)calloc((size_t)STATE_N * WORDS_D, sizeof(uint64_t));
    uint64_t *last = (uint64_t *)calloc((size_t)WORDS_D, sizeof(uint64_t));
    if (!Br || !last) { fprintf(stderr, "Memory allocation failed.\n"); exit(EXIT_FAILURE); }

#define BR(i) (Br + (size_t)(i) * WORDS_D)

    /* Br = A（打包） */
    for (int i = 0; i < STATE_N; i++) {
        uint64_t *bi = BR(i);
        for (int t = 0; t < D_len; t++)
            if (A_mat[i][t]) bi[t >> 6] |= (uint64_t)1 << (t & 63);
    }

    for (int r = 0; r <= BIG_K; r++) {
        uint64_t *row = MROW(r);
        memset(row, 0, (size_t)WORDS_M * sizeof(uint64_t));
        int col = 0;

        for (int b = 0; b < N_PARAM; b++) {
            const uint64_t *ev = EDROW(r + b);
            for (int i = 0; i < STATE_N; i++) {
                if (dot_parity(BR(i), ev, WORDS_D))
                    row[col >> 6] |= (uint64_t)1 << (col & 63);
                col++;
            }
        }

        /* Br <- T * Br，稀疏：行下移一位，再把原末行 XOR 进各 tap 行 */
        if (r < BIG_K) {
            memcpy(last, BR(STATE_N - 1), (size_t)WORDS_D * sizeof(uint64_t));
            for (int i = STATE_N - 1; i >= 1; i--)
                memcpy(BR(i), BR(i - 1), (size_t)WORDS_D * sizeof(uint64_t));
            memset(BR(0), 0, (size_t)WORDS_D * sizeof(uint64_t));
            for (int k = 0; k < TAP_COUNT; k++) {
                uint64_t *bt = BR(LFSR_TAPS[k]);
                for (int w = 0; w < WORDS_D; w++) bt[w] ^= last[w];
            }
        }
    }
#undef BR
    free(Br);
    free(last);
}

/* ============================================================
 * GF(2) 字数组位集工具
 * ============================================================ */

static inline int row_lowest_bit(const uint64_t *row, int w)
{
    for (int k = 0; k < w; k++)
        if (row[k]) return k * 64 + __builtin_ctzll(row[k]);
    return -1;
}

static inline void row_xor(uint64_t *dst, const uint64_t *src, int w)
{
    for (int k = 0; k < w; k++) dst[k] ^= src[k];
}

/* 一般化 RREF，行各占 w 个字，返回主元列数；pivot_cols 长度需 >= ncols */
static int rref_words(uint64_t *rows, int m, int ncols, int w, int *pivot_cols)
{
    int r = 0, npiv = 0;
    for (int col = 0; col < ncols && r < m; col++) {
        int wi = col >> 6;
        uint64_t mask = (uint64_t)1 << (col & 63);
        int pivot = -1;
        for (int i = r; i < m; i++)
            if (rows[(size_t)i * w + wi] & mask) { pivot = i; break; }
        if (pivot < 0) continue;

        if (pivot != r) {
            uint64_t *a = rows + (size_t)r * w;
            uint64_t *b = rows + (size_t)pivot * w;
            for (int k = 0; k < w; k++) { uint64_t tmp = a[k]; a[k] = b[k]; b[k] = tmp; }
        }
        uint64_t *prow = rows + (size_t)r * w;
        for (int i = 0; i < m; i++) {
            if (i == r) continue;
            uint64_t *ri = rows + (size_t)i * w;
            if (ri[wi] & mask) row_xor(ri, prow, w);
        }
        pivot_cols[npiv++] = col;
        r++;
    }
    return npiv;
}

/* ============================================================
 * 单趟增量消元：同时得到 rank(M) 与达到满秩的最小前缀 t
 * （删去原版从不被使用的 coeff 行组合追踪）
 * ============================================================ */

typedef struct { int valid; uint64_t *Mrow; uint64_t *M1row; } Pivot;

static Pivot *pivots = NULL;          /* M_COLS 个，按列索引 */
static uint64_t *pivMstore  = NULL;   /* M_COLS * WORDS_M  */
static uint64_t *pivM1store = NULL;   /* M_COLS * WORDS_M1 */

/* 把 (vM, vM1) 对现有主元做约化；vM 清零返回 1（相关），否则装入新主元返回 0 */
static int reduce_and_maybe_pivot(uint64_t *vM, uint64_t *vM1)
{
    for (;;) {
        int c = row_lowest_bit(vM, WORDS_M);
        if (c < 0) return 1;
        if (pivots[c].valid) {
            row_xor(vM,  pivots[c].Mrow,  WORDS_M);
            row_xor(vM1, pivots[c].M1row, WORDS_M1);
        } else {
            memcpy(pivots[c].Mrow,  vM,  (size_t)WORDS_M  * sizeof(uint64_t));
            memcpy(pivots[c].M1row, vM1, (size_t)WORDS_M1 * sizeof(uint64_t));
            pivots[c].valid = 1;
            return 0;
        }
    }
}

/* 仅约化、不安装主元（用于 build B）；vM 应被约化为 0 */
static void reduce_only(uint64_t *vM, uint64_t *vM1)
{
    for (;;) {
        int c = row_lowest_bit(vM, WORDS_M);
        if (c < 0) return;
        if (pivots[c].valid) {
            row_xor(vM,  pivots[c].Mrow,  WORDS_M);
            row_xor(vM1, pivots[c].M1row, WORDS_M1);
        } else {
            return; /* 满秩后不应发生 */
        }
    }
}

/* 返回最小前缀长度 t，并通过 *full_rank 输出 rank(M) */
static int build_pivots_and_find_t(int *full_rank)
{
    pivots     = (Pivot   *)calloc((size_t)M_COLS, sizeof(Pivot));
    pivMstore  = (uint64_t *)calloc((size_t)M_COLS * WORDS_M,  sizeof(uint64_t));
    pivM1store = (uint64_t *)calloc((size_t)M_COLS * WORDS_M1, sizeof(uint64_t));
    if (!pivots || !pivMstore || !pivM1store) {
        fprintf(stderr, "Memory allocation failed.\n"); exit(EXIT_FAILURE);
    }
    for (int c = 0; c < M_COLS; c++) {
        pivots[c].Mrow  = pivMstore  + (size_t)c * WORDS_M;
        pivots[c].M1row = pivM1store + (size_t)c * WORDS_M1;
    }

    uint64_t *vM  = (uint64_t *)malloc((size_t)WORDS_M  * sizeof(uint64_t));
    uint64_t *vM1 = (uint64_t *)malloc((size_t)WORDS_M1 * sizeof(uint64_t));
    if (!vM || !vM1) { fprintf(stderr, "Memory allocation failed.\n"); exit(EXIT_FAILURE); }

    int rank = 0, t_at_full = 0;
    for (int r = 0; r < M_ROWS; r++) {
        memcpy(vM,  MROW(r),  (size_t)WORDS_M  * sizeof(uint64_t));
        memcpy(vM1, M1ROW(r), (size_t)WORDS_M1 * sizeof(uint64_t));
        if (reduce_and_maybe_pivot(vM, vM1) == 0) {
            rank++;
            t_at_full = r + 1;   /* 1-based 前缀长度 */
        }
    }
    free(vM); free(vM1);

    if (rank == 0) return 0;
    *full_rank = rank;
    return t_at_full;
}

/* ============================================================
 * 构造 B 并求解 Bx = 0
 * ============================================================ */

static void build_B_and_solve(int C, int full_rank, int t)
{
    double step_start;

    if (t + C > M_ROWS || t + C > M1_ROWS) {
        fprintf(stderr, "t + C exceeds the number of matrix rows.");
        exit(EXIT_FAILURE);
    }

    step_start = wall_time_seconds();

    uint64_t *B = (uint64_t *)calloc((size_t)C * WORDS_M1, sizeof(uint64_t));
    if (!B) { fprintf(stderr, "Memory allocation failed."); exit(EXIT_FAILURE); }

    uint64_t *vM  = (uint64_t *)malloc((size_t)WORDS_M  * sizeof(uint64_t));
    uint64_t *vM1 = (uint64_t *)malloc((size_t)WORDS_M1 * sizeof(uint64_t));
    if (!vM || !vM1) { fprintf(stderr, "Memory allocation failed."); exit(EXIT_FAILURE); }

    int check_ok = 1, B_count = 0;
    for (int k = t; k < t + C; k++) {
        memcpy(vM,  MROW(k),  (size_t)WORDS_M  * sizeof(uint64_t));
        memcpy(vM1, M1ROW(k), (size_t)WORDS_M1 * sizeof(uint64_t));
        reduce_only(vM, vM1);
        if (row_lowest_bit(vM, WORDS_M) >= 0) check_ok = 0;
        memcpy(B + (size_t)B_count * WORDS_M1, vM1,
               (size_t)WORDS_M1 * sizeof(uint64_t));
        B_count++;
    }
    free(vM);
    free(vM1);
    (void)full_rank;

    if (!check_ok) {
        free(B);
        fprintf(stderr, "Dependency check in M failed. Cannot solve Bx = 0.");
        exit(EXIT_FAILURE);
    }
    print_step_time("Build B / dependency reduction", step_start);

    step_start = wall_time_seconds();

    uint64_t *R = (uint64_t *)malloc(
        (size_t)B_count * WORDS_M1 * sizeof(uint64_t));
    if (!R) {
        free(B);
        fprintf(stderr, "Memory allocation failed.");
        exit(EXIT_FAILURE);
    }
    memcpy(R, B, (size_t)B_count * WORDS_M1 * sizeof(uint64_t));

    int *pivot_cols = (int *)malloc((size_t)M1_COLS * sizeof(int));
    char *is_pivot  = (char *)calloc((size_t)M1_COLS, 1);
    int *free_cols  = (int *)malloc((size_t)M1_COLS * sizeof(int));
    if (!pivot_cols || !is_pivot || !free_cols) {
        fprintf(stderr, "Memory allocation failed.");
        exit(EXIT_FAILURE);
    }

    int npiv = rref_words(R, B_count, M1_COLS, WORDS_M1, pivot_cols);
    for (int i = 0; i < npiv; i++)
        is_pivot[pivot_cols[i]] = 1;

    int nfree = 0;
    for (int j = 0; j < M1_COLS; j++)
        if (!is_pivot[j])
            free_cols[nfree++] = j;

    print_step_time("RREF(B)", step_start);

    step_start = wall_time_seconds();

    int nbasis = nfree;
    uint64_t *basis = (uint64_t *)calloc(
        (size_t)(nbasis > 0 ? nbasis : 1) * WORDS_M1, sizeof(uint64_t));
    if (!basis) {
        fprintf(stderr, "Memory allocation failed.");
        exit(EXIT_FAILURE);
    }

    for (int fi = 0; fi < nfree; fi++) {
        int free_column = free_cols[fi];
        uint64_t *x = basis + (size_t)fi * WORDS_M1;
        x[free_column >> 6] |= (uint64_t)1 << (free_column & 63);

        for (int ri = 0; ri < npiv; ri++) {
            if ((R[(size_t)ri * WORDS_M1 + (free_column >> 6)]
                 >> (free_column & 63)) & 1u) {
                int pc = pivot_cols[ri];
                x[pc >> 6] |= (uint64_t)1 << (pc & 63);
            }
        }
    }

    print_step_time("Construct Ker(B) basis", step_start);

    step_start = wall_time_seconds();

    printf("Solution of Bx = 0:\n");
    printf("\nrank(B) = %d", npiv);
    printf("\ndim Ker(B) = %d", nbasis);

    if (nbasis > 0) {
        printf("\nBasis of Ker(B):");
        for (int i = 0; i < nbasis; i++) {
            uint64_t *x = basis + (size_t)i * WORDS_M1;
            printf("v%d = [", i + 1);
            for (int j = 0; j < M1_COLS; j++)
                printf("%s%d", j ? ", " : "",
                       (int)(x[j >> 6] >> (j & 63) & 1u));
            printf("]");
        }

        {
            uint64_t *x = basis;
            printf("\nOne nonzero solution x:[");
            for (int j = 0; j < M1_COLS; j++)
                printf("%s%d", j ? ", " : "",
                       (int)(x[j >> 6] >> (j & 63) & 1u));
            printf("]");

            FILE *xfp = fopen("x.txt", "w");
            if (xfp == NULL) {
                perror("Cannot open x.txt");
                exit(EXIT_FAILURE);
            }

            fprintf(xfp, "[");
            for (int j = 0; j < M1_COLS; j++) {
                int value = (int)((x[j >> 6] >> (j & 63)) & 1u);
                fprintf(xfp, "%s%d", j ? ", " : "", value);
            }
            fprintf(xfp, "]\n");

            if (fclose(xfp) != 0) {
                perror("Cannot close x.txt");
                exit(EXIT_FAILURE);
            }

            printf("Wrote %d bits to x.txt\n", M1_COLS);
        }
    } else {
        printf("Only the zero solution exists.");
    }

    fflush(stdout);
    print_step_time("Print results / write x.txt", step_start);

    free(basis);
    free(free_cols);
    free(is_pivot);
    free(pivot_cols);
    free(R);
    free(B);
}

/* ============================================================
 * 主函数
 * ============================================================ */

int main(void)
{
    const double program_start = wall_time_seconds();
    double step_start;
    const char *polynomial_file = POLY_FILE;
    int C = C_PARAM;
    int full_rank, t;
    int L12, Q12;

    printf("========== Step timing ==========\n");

    step_start = wall_time_seconds();
    if (!read_support_set(polynomial_file)) {
        fprintf(stderr, "Failed to read or parse file %s.\n", polynomial_file);
        return EXIT_FAILURE;
    }
    print_step_time("Read F(x) support set", step_start);

    step_start = wall_time_seconds();
    if (!allocate_sequences()) {
        fprintf(stderr, "Failed to allocate sequence buffers.\n");
        return EXIT_FAILURE;
    }
    print_step_time("Allocate sequence buffers", step_start);

    step_start = wall_time_seconds();
    if (!generate_transition_matrix()) {
        fprintf(stderr, "Failed to generate transition matrix T.\n");
        return EXIT_FAILURE;
    }
    print_step_time("Generate transition matrix T", step_start);

    step_start = wall_time_seconds();
    if (!generate_A_matrix()) {
        fprintf(stderr, "Failed to generate matrix A.\n");
        return EXIT_FAILURE;
    }
    print_step_time("Generate matrix A", step_start);

    printf("F(x) support set loaded successfully: D_len = %d, max degree = %d\n",
           D_len, D[D_len - 1]);
    printf("Tight sequence length = %d (OMAX = %d)\n", seq_len, OMAX);

    step_start = wall_time_seconds();
    generate_sequences();
    print_step_time("Generate e/sprime sequences", step_start);

    step_start = wall_time_seconds();
    if (!build_D_precompute()) {
        fprintf(stderr, "Failed to allocate D-precompute buffers.\n");
        return EXIT_FAILURE;
    }
    print_step_time("Build D precompute (eD/pe/Q)", step_start);

    step_start = wall_time_seconds();
    M_bits  = (uint64_t *)calloc(
        (size_t)M_ROWS * WORDS_M, sizeof(uint64_t));
    M1_bits = (uint64_t *)calloc(
        (size_t)M1_ROWS * WORDS_M1, sizeof(uint64_t));

    if (!M_bits || !M1_bits) {
        fprintf(stderr, "Memory allocation failed.\n");
        return EXIT_FAILURE;
    }
    print_step_time("Allocate M / M1", step_start);

    step_start = wall_time_seconds();
    build_M1();
    print_step_time("Build M1", step_start);

    step_start = wall_time_seconds();
    build_M();
    print_step_time("Build M", step_start);

    step_start = wall_time_seconds();
    L12 = Lprime_fn(1, 2);
    Q12 = reduced_quadratic_fn(1, 2);
    printf("L'_{1,2} = %d\n", L12);
    printf("reduced quadratic (1,2) = %d\n", Q12);
    print_step_time("Compute L' and quadratic", step_start);

    step_start = wall_time_seconds();
    t = build_pivots_and_find_t(&full_rank);
    if (t == 0) {
        fprintf(stderr, "rank(M) is 0. Cannot continue.\n");
        return EXIT_FAILURE;
    }
    print_step_time("Incremental elimination / rank(M)", step_start);

    printf("Number of columns of M = %d\n", M_COLS);
    printf("Actual rank of M = %d\n", full_rank);
    printf("Full-rank prefix length t = %d\n", t);

    build_B_and_solve(C, full_rank, t);

    step_start = wall_time_seconds();
    free(pivM1store);
    free(pivMstore);
    free(pivots);
    free(M1_bits);
    free(M_bits);
    free(Qtab);
    free(pe);
    free(eD);
    free(sprime);
    free(e_seq);
    print_step_time("Free allocated memory", step_start);

    fflush(stdout);
    printf("=================================\n");
    printf("[TIME] %-30s : %.6f seconds\n",
           "TOTAL", wall_time_seconds() - program_start);

    return EXIT_SUCCESS;
}
