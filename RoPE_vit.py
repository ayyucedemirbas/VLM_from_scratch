import tensorflow as tf
import math
from functools import partial

def init_2d_freqs(dim, num_heads, theta=10.0, rotate=True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (tf.range(0, dim, 4, dtype=tf.float32)[: (dim // 4)] / dim))
    for i in range(num_heads):
        angles = tf.random.uniform([1], maxval=2 * math.pi) if rotate else tf.zeros([1])
        fx = tf.concat([mag * tf.cos(angles), mag * tf.cos(math.pi/2 + angles)], axis=-1)
        fy = tf.concat([mag * tf.sin(angles), mag * tf.sin(math.pi/2 + angles)], axis=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = tf.stack(freqs_x, axis=0)
    freqs_y = tf.stack(freqs_y, axis=0)
    freqs = tf.stack([freqs_x, freqs_y], axis=0)
    return freqs

def init_t_xy(end_x, end_y):
    t = tf.range(end_x * end_y, dtype=tf.float32)
    t_x = (t % end_x)
    t_y = tf.floor(t / end_x)
    return t_x, t_y

def compute_mixed_cis(freqs, t_x, t_y, num_heads):
    N = tf.shape(t_x)[0]
    freqs_x = tf.einsum('i,jk->jik', t_x, freqs[0])  # shape [N, num_heads, dim]
    freqs_y = tf.einsum('i,jk->jik', t_y, freqs[1])  # shape [N, num_heads, dim]
    freqs_cis = tf.complex(tf.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis

def compute_axial_cis(dim, end_x, end_y, theta=100.0):
    freqs_x = 1.0 / (theta ** (tf.range(0, dim, 4, dtype=tf.float32)[: (dim // 4)] / dim))
    freqs_y = 1.0 / (theta ** (tf.range(0, dim, 4, dtype=tf.float32)[: (dim // 4)] / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = tf.einsum('i,j->ij', t_x, freqs_x)
    freqs_y = tf.einsum('i,j->ij', t_y, freqs_y)
    freqs_cis_x = tf.complex(tf.ones_like(freqs_x), freqs_x)
    freqs_cis_y = tf.complex(tf.ones_like(freqs_y), freqs_y)
    return tf.concat([freqs_cis_x, freqs_cis_y], axis=-1)

def reshape_for_broadcast(freqs_cis, x):
    ndim = len(x.shape)
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
    return tf.reshape(freqs_cis, shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = tf.complex(xq[..., 0::2], xq[..., 1::2])
    xk_ = tf.complex(xk[..., 0::2], xk[..., 1::2])
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = tf.concat([tf.math.real(xq_ * freqs_cis), tf.math.imag(xq_ * freqs_cis)], axis=-1)
    xk_out = tf.concat([tf.math.real(xk_ * freqs_cis), tf.math.imag(xk_ * freqs_cis)], axis=-1)
    return xq_out, xk_out

class Attention(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)

    def call(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = tf.einsum('...ij,...kj->...ik', q, k)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.einsum('...ij,...jk->...ik', attn, v)
        x = tf.reshape(x, (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class RoPEAttention(Attention):
    """Multi-head Attention block with rotary position embeddings."""
    def __init__(self, *args, rope_theta=10.0, rope_mixed=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.rope_mixed = rope_mixed

        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)

            freqs = init_2d_freqs(
                dim=self.dim // self.num_heads, num_heads=self.num_heads, theta=rope_theta,
                rotate=True
            )
            self.freqs = tf.Variable(freqs, trainable=True)

            t_x, t_y = init_t_xy(end_x=14, end_y=14)
            self.freqs_t_x = tf.convert_to_tensor(t_x)
            self.freqs_t_y = tf.convert_to_tensor(t_y)
        else:
            self.compute_cis = partial(compute_axial_cis, dim=self.dim // self.num_heads, theta=rope_theta)
            freqs_cis = self.compute_cis(end_x=14, end_y=14)
            self.freqs_cis = tf.convert_to_tensor(freqs_cis)

    def call(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        w = h = tf.sqrt(tf.cast(N - 1, tf.float32))
        if self.rope_mixed:
            t_x, t_y = self.freqs_t_x, self.freqs_t_y
            if tf.shape(self.freqs_t_x)[0] != N - 1:
                t_x, t_y = init_t_xy(end_x=w, end_y=h)
                t_x, t_y = tf.convert_to_tensor(t_x), tf.convert_to_tensor(t_y)
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
        else:
            freqs_cis = self.freqs_cis
            if tf.shape(self.freqs_cis)[0] != N - 1:
                freqs_cis = self.compute_cis(end_x=w, end_y=h)
            freqs_cis = tf.convert_to_tensor(freqs_cis)


        q_rot, k_rot = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)

        q = tf.concat([q[:, :, :1], q_rot], axis=2)
        k = tf.concat([k[:, :, :1], k_rot], axis=2)

        attn = (q * self.scale) @ tf.transpose(k, perm=[0, 1, 3, 2])
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)


        x = tf.reshape(x, (B, N, self.num_heads, C // self.num_heads))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
     
