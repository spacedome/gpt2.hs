module Model where

import Data.List (transpose)
import Numeric.LinearAlgebra (Matrix, Vector, build, cmap, rows, scalar, size, sumElements, tr, (#>), (<>))
import Numeric.LinearAlgebra.Data (fromColumns, fromRows, takesV, toRows, vjoin)
import Prelude hiding ((<>))

-- handy type aliases

type Token = Int

type V = Vector Float

type M = Matrix Float

-- layer types

newtype TokenEmbedding = TokenEmbedding [V]

newtype PositionEmbedding = PositionEmbedding [V]

data Block = Block LayerNorm Attention LayerNorm MLP

data MLP = MLP M V M V

data LayerNorm = LayerNorm V V

data Attention = Attention M V M V

class Layer a where
  forward :: a -> [V] -> [V]

data GPT = GPT
  { wpe :: PositionEmbedding,
    wte :: TokenEmbedding,
    blocks :: [Block],
    lnf :: LayerNorm
  }

-- plain old softmax. inlining didn't seem to help much
softmax :: V -> V
softmax v = expv * scalar (1 / sumElements expv)
  where expv = cmap exp v

-- approximation of gelu using tanh (close enough approx to not matter)
gelu :: V -> V
gelu x = 0.5 * x * (1 + tanh (sqrt (2 / pi) * (x + 0.044715 * x * x * x)))

-- Function to fill the upper triangular part of a matrix with -inf.
-- inside the attention head this has the effect of making tokens
-- only depend on previous tokens
tril :: Int -> M
tril n = build (n, n) (\i j -> if j > i then -1 / 0 else 0)

-- the model combines a token indexed and position indexed embedding 
embedding :: TokenEmbedding -> PositionEmbedding -> [Token] -> [V]
embedding (TokenEmbedding te) (PositionEmbedding pe) ts =
  zipWith (+) (fmap (te !!) ts) pe

-- layernorm is a simple norm, just make sure you do it in the right dim.
-- not needing to handle the backward pass makes this not so bad.
instance Layer LayerNorm where
  forward layer = fmap (forwardLN layer)
    where
      forwardLN :: LayerNorm -> V -> V
      forwardLN (LayerNorm w b) x = y
        where
          n = fromIntegral (size x)
          mean = scalar (sumElements x / n)
          cent = x - mean
          varx = sumElements (cent * cent) / n
          fact = scalar (sqrt (varx + 1e-5))
          y = ((x - mean) / fact) * w + b

-- the first part of the attention head is a linear layer.
-- Q,K,V weights and heads are combined and we have to take them apart here.
attnAtToken :: Attention -> V -> ([V], [V], [V])
attnAtToken (Attention w b _ _) x = (qh, kh, vh)
  where
    y = (w #> x) + b
    -- split apart into Q, K, V components
    qkv = takesV [768, 768, 768] y
    -- probably a better way to do this lol
    (q, k, v) = (head qkv, (head . tail) qkv, (head . tail . tail) qkv)
    -- split into individual heads
    qh = takesV (replicate 12 64) q
    kh = takesV (replicate 12 64) k
    vh = takesV (replicate 12 64) v

-- this is the actual attention part where we construct the attention matrix.
attnHead :: (M, M, M) -> M
attnHead (q, k, v) = z
  where
    attnMatrix = tr q <> k * scalar (1 / 8) -- 1 / sqrt (size k)
    -- mask the upper right triangular to -inf (becomes 0 in softmax)
    attnMasked = tril (rows attnMatrix) + attnMatrix
    -- no tensor library means we have to do this kinda stuff
    attnSoftmax = fromRows (fmap softmax (toRows attnMasked))
    z = attnSoftmax <> tr v

instance Layer Attention where
  forward at@(Attention _ _ w b) xs = z
    where
      (q, k, v) = unzip3 (fmap (attnAtToken at) xs)
      qh = fmap fromColumns (transpose q)
      kh = fmap fromColumns (transpose k)
      vh = fmap fromColumns (transpose v)
      lm = fmap attnHead (zip3 qh kh vh)
      y = fmap vjoin (transpose (fmap toRows lm))
      z = fmap ((+ b) . (w #>)) y

-- just a normal multi layer perceptron layer with a gelu nonlinearity
instance Layer MLP where
  forward (MLP wfc bfc wproj bproj) x = x3
    where
      x1 = fmap ((+ bfc) . (wfc #>)) x
      x2 = fmap gelu x1
      x3 = fmap ((+ bproj) . (wproj #>)) x2

-- the transformer blocks combine altenating attention heads and MLPs
-- with layernorms and passthrough
instance Layer Block where
  forward (Block l1 at l2 mp) xs = x4
    where
      x1 = forward l1 xs
      x2 = zipWith (+) xs (forward at x1)
      x3 = forward l2 x2
      x4 = zipWith (+) x2 (forward mp x3)

-- takes in sequence of tokens, outputs logits to be sampled
forwardModel :: GPT -> [Token] -> [V]
forwardModel model tokens = x3
  where
    TokenEmbedding wtew = wte model
    emb = embedding (wte model) (wpe model) tokens
    x1 = foldr forward emb (blocks model)
    x2 = forward (lnf model) x1
    x3 = fmap (tr (fromColumns wtew) #>) x2
