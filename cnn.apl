⎕ml←3
⎕io←0

⍝ Get the number of elements first, and then reduce with + and divide by n.
avg ← { (+/÷≢),⍵ }

⍝ Maxpos of the array of shape [10,1,1,1,...] where values are either 0 or 1
⍝ First elements of the indices of the array sorted in the descendingn order.
maxpos ← {(,⍵)⍳⌈/,⍵}
⍝ Alternative: { (⍸,⍵)[0] }


⍝ Convolve with an array `a` and weights `w` looks like this:
⍝ So `⍳⍴w`                  is a matrix of all the indices of `w`,
⍝     {(1+(⍴a)-⍴w)↑⍺↓⍵}     computes `a` shifted by ⍺ and then takes
⍝                           the remaining shape
⍝ Then we multiply enclosed arrays with `w`, sum them up and disclose them
⍝

⍝ conv←{s ← 1+(⍴⍵)-⍴⍺ ⋄ ⊃+/,⍺×(⍳⍴⍺){s↑⍺↓⍵}¨⊂⍵}
⍝ conv←{a←⍵ ⋄ s←1+(⍴a)-⍴⍺ ⋄ ⊃+/,⍺×{s↑⍵↓a}¨⍳⍴⍺}
⍝ conv←{s ← 1+(⍴⍵)-⍴⍺ ⋄ ∧/(⍴⍺)=⍴⍵ : s⍴+/,⍺×⍵ ⋄ ⊃+/,⍺×(⍳⍴⍺){s↑⍺↓⍵}¨⊂⍵}
⍝ conv←{a ← ⍵ ⋄ s ← 1+(⍴a)-⍴⍺ ⋄ ∧/(⍴⍺)=⍴a : s⍴+/,⍺×a ⋄ ⊃+/,⍺{⍺×s↑⍵↓a}¨⊂⍳⍴⍺}
conv←{a←⍵ ⋄ s←1+(⍴a)-⍴⍺ ⋄ ∧/(⍴⍺)=⍴a : s⍴+/,⍺×a ⋄  ⊃+/,⍺×{s↑⍵↓a}¨⍳⍴⍺}


⍝ Here is a version without the enclose/disclose and it got slower
⍝ and uglier.  Not sure whether there is an easy fix.
⍝
⍝ conv ← { w←⍺ ⋄ a←⍵ ⋄ s ← 1+(⍴a)-⍴w ⋄ +⌿((×/⍴w),s)⍴w{⍺×⍵}⍤(0,(⍴⍴w))⊣{s↑⍵↓a}⍤1⊣⊃⍳⍴w }
⍝
⍝ Here we have an alternative version of the conv that uses the ⌺ operator
⍝ However, it is noticeably slower in our application, as we have to remove
⍝ the "shrads" around the actual stencil operation.
⍝
⍝ conv←{w←⍺ ⋄ a←⍵ ⋄ t ← ⌊2÷⍨(⍴w)-~2|⍴w ⋄ (-t)↓t↓ ({+/,w×⍵}⌺(⍴w)⊢a)}


⍝ Multiconv is just a rank operator:
⍝ For weights `ws` with dimensionality greater than the dimensionality of `a`
⍝ Here we assume that `bias` has the same dimensionality as the "outer" level
⍝ of `ws` and it contains scalars.  (If not this can be easily adapted).
⍝
⍝ multiconv← {(a ws bias)←⍵ ⋄ ⊃bias+{⊂⍵ conv a}⍤(⍴⍴a)⊣ws}
multiconv← {(a ws bias)←⍵ ⋄ bias{⍺ + ⍵ conv a}⍤(0,(⍴⍴a))⊣ws}


⍝ Here we can pre-optimise conv for FC layer, but it doesn't really help
⍝
⍝ fcconv←{s ← 1+(⍴⍵)-⍴⍺ ⋄ s⍴+/,⍺×⍵ }
⍝ fclayer← {(a ws bias)←⍵ ⋄ bias{⍺ + ⍵ fcconv a}⍤(0,(⍴⍴a))⊣ws}


⍝ Simply the sum
backbias ← {+/,⍵}


⍝ For every index iv of `w`, we compute w[iv]*d_out and shift it
⍝ by iv.  Then we take shape(in) elements of that and sum all such
⍝ arrays with `+`.
⍝
backin ← {(d w in)←⍵ ⋄ ⊃+/,w{(⍴in)↑(-⍵+⍴d)↑⍺×d}¨⍳⍴w}


⍝ Logistic function is 1/(1+e^(-in))
logistic ← { 1÷1+ *-⍵ }

⍝ Mean Squared Error
meansqerr ← {÷∘2+/,(⍺-⍵)*2}


⍝ Backlogistic is a simple expression
backlogistic ← {⍺×⍵×1-⍵}



⍝ This looks a bit ugly as we have to account for applying avgpool
⍝ over the last (dim f) dimensions of a.  Therefore we use rank operator
⍝ twice.  Other than that we simply reshape the array into (|a|/f)++f
⍝ and apply avg.
⍝ avgp ← {a ← ⍵ ⋄ {(i j)←2×⍵ ⋄ avg ⊣ ((i (i+1))(j (j+1))) ⌷ a} ¨⍳(÷∘2⍴a)}
avgpool ← {(x y)←⍴⍵ ⋄ avg⍤2⊣(x÷2)(y÷2)2 2⍴⍉⍤2⊣(x÷2)2 y⍴⍵}⍤2

⍝ This is a slower version, but it is very concise.
⍝ avgpool1←{÷∘4{+/,⍵}⌺(2 2⍴2)⍤2⊣⍵}



⍝ For every element in `a` compute an array of shape `f` where all the
⍝ elements are the current element of `a` divided (prod f).
⍝ backavgpool ← {(a f) ← ⍵ ⋄ ⊃⍪/{⊂⍵}⍤2⊣⊃,/{f⍴⍵÷×/f}¨a }

⍝ We are tempted to specialise average pool for the shape 2 2, as we
⍝ don't use anything else and the expression gets really nice
backavgpool  ← {2⌿2/⍵÷4}⍤2


⍝ Convert the label into this stupid format
convlab ← { 10 1 1 1 1 ⍴ ⍵=⍳10 }

⍝ Some IO functions
Ubyte ← {⍵+256×⍵<0}

∇ GetFileInt8←{
    ⎕←'Reading file: ', ⍵ ,' as int8'
    ntn←⍵ ⎕NTIE 0 ⋄ z←⎕NREAD ntn,83 ¯1 ⋄ ntn←⎕NUNTIE ntn ⋄ z
}
∇

GetInt←{f←Ubyte ⍵ ⋄ 256⊥⍤1⊢(((⍴f)÷4),4)⍴f}

∇ ReadImages←{
    t←GetFileInt8 ⍵
    z←(GetInt 12↑4↓t)⍴16↓t
    ⎕←'Read ',(⍕⍴z),'images from ',⍵
    z
 }
∇

∇ ReadLabels←{
    z←8↓GetFileInt8 ⍵
    ⎕←'Read ',(⍕⍴z),' labels from ',⍵
    z
 }
∇


⍝ backmulticonv is just a ranked application of backin, backw and backbias.
∇ backmulticonv ← {
  (d_out weights in bias) ← ⍵
  d_in ← +⌿d_out {backin ⍺ ⍵ in} ⍤((⍴⍴in), (⍴⍴in)) ⊣ weights
  d_w ← {⍵ conv in} ⍤(⍴⍴in) ⊣ d_out
  d_bias ← backbias ⍤(⍴⍴in) ⊣ d_out
  d_in d_w d_bias
}
∇


∇ trainzhang ← {
  (img target k1 b1 k2 b2 fc b) ← ⍵
  c1 ← logistic multiconv img k1 b1
  s1 ← avgpool c1
  c2 ← logistic multiconv s1 k2 b2
  s2 ← avgpool c2
  out ← logistic multiconv s2 fc b
  d_out ← out - target
  err ← out  meansqerr target

  (d_s2 d_fc d_b) ← backmulticonv (d_out backlogistic out) fc s2 b
  d_c2 ← backavgpool d_s2
  bl1 ← d_c2 backlogistic c2
  (d_s1 d_k2 d_b2) ← backmulticonv bl1 k2 s1 b2
  d_c1 ← backavgpool d_s1
  (_ d_k1 d_b1) ← backmulticonv (d_c1 backlogistic c1) k1 img b1
  (d_k1 d_b1 d_k2 d_b2 d_fc d_b err)
}
∇

∇ testzhang ← {
  (k1 b1 k2 b2 fc b) ← ⍵
  c1 ← logistic multiconv ⍺ k1 b1
  s1 ← avgpool c1
  c2 ← logistic multiconv s1 k2 b2
  s2 ← avgpool c2
  out ← logistic multiconv s2 fc b
  (maxpos out)
}
∇


∇ train ← {
    (e i k1 b1 k2 b2 fc b rate imgs labs trsz) ← ⍵

    (i ≥ trsz) : (e÷trsz) k1 b1 k2 b2 fc b
    img ← i ⌷ imgs
    target ← convlab ⊣ i ⌷ labs
    (d_k1 d_b1 d_k2 d_b2 d_fc d_b err) ← trainzhang (img target k1 b1 k2 b2 fc b)

    k1←k1-rate×d_k1
    k2←k2-rate×d_k2
    b1←b1-rate×d_b1
    b2←b2-rate×d_b2
    fc←fc-rate×d_fc
    b← b -rate×d_b
    error←+/err
    ∇ ((e+error) (i+1)  k1 b1 k2 b2 fc b rate imgs labs trsz)
}
∇


∇ main ← {
  epochs    ← 10
  ⍝ batchsize ← 100
  trainings ← 1000
  tests     ← 10000
  rate      ← 0.05
  k1        ← 6 5 5⍴÷25
  b1        ← 6⍴÷6
  k2        ← 12 6 5 5⍴÷150
  b2        ← 12⍴÷12
  fc        ← 10 12 1 4 4⍴÷192
  b         ← 10⍴÷10
  trimgs    ← ReadImages 'input/train-images-idx3-ubyte'
  teimgs    ← ReadImages 'input/t10k-images-idx3-ubyte'
  trlabs    ← ReadLabels 'input/train-labels-idx1-ubyte'
  telabs    ← ReadLabels 'input/t10k-labels-idx1-ubyte'

  ⎕←'Running Zhang with ',(⍕epochs),' epochs, batchsize ',(⍕batchsize)
  ⎕←(⍕trainings),' training images, ',(⍕tests),' tests',' and a rate of ',⍕rate

  (k1 b1 k2 b2 fc b) ← {
    t←⎕AI
    (e k1 b1 k2 b2 fc b) ← train (0 0), ⍵, (rate trimgs trlabs trainings)
    ⎕←'The time taken for training is ',(⍕⎕AI-t)
    ⎕←'The average error after training is ',(⍕e)
    k1 b1 k2 b2 fc b
  }⍣epochs ⊣ (k1 b1 k2 b2 fc b)

  t←⎕AI ⋄ correct←+/telabs = teimgs testzhang⍤2 ⊣(k1 b1 k2 b2 fc b)
  ⎕←'The time taken for recognition is ',(⍕⎕AI-t)
  ⎕←(⍕correct),' images out of ',(⍕tests),' recognised correctly'
}
∇

main 0


