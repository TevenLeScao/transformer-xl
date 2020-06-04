import copy
import numpy as np

from conversions import day_ratio


def clean_run(run):
    return [(a, float(b)) for a, b in run if b != "undefined"]


def param_count(run):
    compute_per_eval = run[0][0]
    return round(compute_per_eval / 4000 / 150 / 60 / 6 * day_ratio)


def convert_to_logspace(run, a, b, c):
    logspace_run = copy.deepcopy(run)
    logspace_run[:, 0] = b * np.log(run[:, 0])
    logspace_run[:, 1] = -np.log(run[:, 1] - c) + np.log(a)
    return logspace_run


# OpenAI used another unit for floating-point operations with a ratio of the number of seconds in a day; we'll display
# the raw number, but do the calculations with the ratio as it can overflow without it (convex hull notably fails)


def hf_code(width, depth):

    return f"""<span style="color: #008800; font-weight: bold">import</span> <span style="color: #0e84b5; font-weight: bold">transformers</span>
config <span style="color: #333333">=</span> transformers<span style="color: #333333">.</span>TransfoXLConfig(d_model<span style="color: #333333">=</span><span style="color: #40a070">{width}</span>, d_embed<span style="color: #333333">=</span><span style="color: #40a070">{width}</span>, n_head<span style="color: #333333">=</span><span style="color: #40a070">8</span>, d_head<span style="color: #333333">=</span><span style="color: #40a070">{int(width / 8)}</span>, d_inner<span style="color: #333333">=</span><span style="color: #40a070">{width}</span>, n_layer<span style="color: #333333">=</span><span style="color: #40a070">{depth}</span>, tgt_len<span style="color: #333333">=</span><span style="color: #40a070">152</span>, mem_len<span style="color: #333333">=</span><span style="color: #40a070">152</span>)
 model <span style="color: #333333">=</span> transformers<span style="color: #333333">.</span>TransfoXLModel(config)"""


def kWh_fill(kWh):
    return 'This will consume around <span style="font-weight: bold">{:.0f}</span> kWh'.format(kWh)


def co2_fill(co2):
    return 'And release <span style="font-weight: bold">{:.2f}</span> tons of CO2.'.format(co2)

md1 = """<h1 id="how-big-should-my-language-model-be-">How Big Should My Language Model Be ?</h1>
<p>Natural Language Processing can sometimes feel like model size is optimized for headlines. <a href="https://arxiv.org/abs/2005.14165">175 billion parameters</a> is certainly an eye-catching number! Why not just train more efficiently with a smaller model? One surprising scaling effect of deep learning is that <strong>bigger neural networks are actually compute-efficient.</strong> This is something OpenAI in particular has explored in papers like <em><a href="https://arxiv.org/abs/2001.08361">Scaling Laws for Neural Language Models</a></em>. Research at Hugging Face also leverages this phenomenon, and we&#39;ve combined it with GPU speed estimations to ensure model size is just right for the compute budget of the experiment (when in doubt, it&#39;s bigger than you think!). This blog post will show how this impacts architecture decisions on a standard language modeling benchmark. <a name="start">Wanna</a> play with our demo before reading? Just click <a href="#demo">here</a>!</p>
<h2 id="1-there-is-an-optimal-time-to-stop-training-and-its-earlier-than-you-think">1. There is an optimal time to stop training and it&#39;s earlier than you think!</h2>
<p>Let&#39;s look at some loss curves. For our example, the task will be training <a href="https://arxiv.org/pdf/1901.02860.pdf">Transformer-XL</a>, the state-of-the-art in language modeling, on Wikitext-103, a standard, medium-size benchmark. As training progresses, we&#39;ll look at the performance of the model (as measured by validation loss) depending on compute cost (as measured by floating point operations). Let&#39;s run a few experiments! In the following plot, every line of colour corresponds to a Transformer-XL run of 200000 steps with a different number and size of layers, with all other hyperparameters kept the same. This spans models from a mere thousand to a hundred million parameters (excluding embeddings). Bigger models are on the right as they require more compute for every step. Don&#39;t worry, we&#39;ve already run them so you don&#39;t have to.</p>"""
md2 = """
<p>As introduced in <em>Scaling Laws</em>, we plot the validation loss against non-embedding floating-point operations (neFLOs). There seems to be a frontier of performance for a given neFLO budget that no model manages to beat, depicted here in red. In <em>Scaling Laws</em>, it is referred to as the compute frontier. Every run reaches it, or comes close, after an initial phase of quick loss improvement, then tapers away as the end of training is not as efficient. This has a very practical implication: if you have a certain budget in floating-point operations, to reach the best performance, you should choose a model size that reaches the compute frontier after that many operations and stop it at that moment. This is actually way before model convergence, which usually happens around 10 times later! For example, <strong>the CMU state-of-the-art result ran for 4 million steps even though that model stopped being optimal at around 500 000.</strong> Starting from an even bigger model and stopping earlier would have yielded a better loss for that (huge) compute budget.</p>
<p>In <em>Scaling Laws</em>, the OpenAI team fitted a power law to the compute frontier on GPT-2 training. This still seems to be a good fit in our task. In addition, we also fitted a power law between the budget budget and the number of parameters of the model that is optimal for that budget. It is pictured in the following plot.</p>

"""
md3 = """
<p>As good-performing models tend to spend considerable time tangent on the compute frontier, there is a bit of noise in the relationship. However, this also means that there is more tolerance in the estimation even if the model size we predict is a bit off, as the imperfect model will still be very close to optimal.</p>
<h2 id="2-gpus-are-optimized-for-large-wide-models">2. GPUs are optimized for large, wide models</h2>
<p>Not that we have established a rule connecting performance, optimal size, and neFLOs, we need to translate it into a more actionable metric. Whether you are constrained by temporal or financial constraints, the main resource is GPU time. In order to establish a connection between neFLOs and GPU time, we benchmarked different Transformer-XL model sizes on 4 different GPUs available on Google Cloud Platform across tens of thousands of runs, taking into account mixed precision training. Here are our findings:</p>
<h5 id="speed-model">Speed model</h5>
<img class='center' style='height: 1.25em;' src='optimal_training/static/formula_1.png' alt='formula_1'>
<p>neFLOs per second speed can be modeled as a factorized multivariate function of model width (the number of neurons per layer), depth (the number of layers) and batch size, by increasing order of importance. <strong>The maximum prediction error was 15% of the observed speed.</strong></p>
<h5 id="width">Width</h5>
<p>GPUs are optimized for the large feed-forward layers of wide transformers. In all of our experiments, neFLOs per second depended on model width as <strong>a power law of exponent around 1.6</strong>. This means that a model that&#39;s twice as wide, which requires 4 times more operations, also goes through those operations around 3.16 times faster, <strong>nearly offsetting the additional compute cost</strong>.</p>
<h5 id="depth">Depth</h5>
<p>neFLOs per second were also positively correlated with depth. Our best results were attained by modeling this connection as proportional to depth * (depth + additive constant). This is coherent with the fact that Transformers must process layers serially. In essence, <strong>deeper models aren&#39;t actually faster, but they appear to be so as their overhead is smaller relative to the more productive operations</strong>. The additive constant, which represents this overhead, was consistently around 5 in our experiments, which essentially means that data loading to the GPU, embeddings, and softmax operations, represent around 5 transformer layers&#39; worth of time.</p>
<h5 id="batch-size">Batch size</h5>
<p>Batch size played the least role. It was <strong>positively correlated with speed for small values, but quickly saturated</strong> (and even seemed to hurt at high values, after 64 on the V100 and P100 and 16 on the K80 and P4). We modeled its contribution as a logarithmic function to keep things simple as it was also the variable for which the factorized independence assumption was the weakest. We ran all our experiments at size 64 on a single GPU. This is another perk of big models: <strong>as bigger batch sizes don&#39;t seem to help much, if your model is too big to fit on a GPU, you could just use a smaller batch size and gradient accumulation.</strong></p>
<h5 id="powers-of-2-still-matter-in-2020">Powers of 2 still matter in 2020!</h5>
<p>Finally, one surprising takeaway was that <strong>hyperparameters whose width or batch size were powers of 2 out-performed the others</strong>. That was the case both using <code>torch.backends.cudnn.benchmark</code> to speed up training on the V100 (this requires multiples of 8) and on other GPUs. In the end, we only fitted on powers of 2 as fitting on all data points meant a poor fit quality that consistently under-estimated speed for powers of 2 points, and one might as well choose the fastest parameters.</p>
<p>In the end, our final estimation of operation speed was as follows:</p>
<img class='center' style='height: 2.5em;' src='optimal_training/static/formula_2.png' alt='formula_2'>
<p>with, for example on a V100 GPU without mixed precision, k=2.21*10<sup>7</sup>, a=1.66, b=5.92, and c=1.33. Different GPUs had close results with a different multiplicative constant.</p>
<h2 id="3-demonstration-on-a-language-modeling-task-wikitext-103">3. <a name="demo">Demonstration on a language modeling task: Wikitext-103</a></h2>
<p>Now that we have a relation between model size and training speed, we can predict, for a certain GPU time or price budget, the optimal model size on the task and the performance it will achieve.</p>

"""
md4 = """<p>Prices are indicated for Google Cloud Platform. The energy consumption was estimated thanks to Peter Henderson&#39;s <a href="https://github.com/Breakend/experiment-impact-tracker">Experiment impact tracker</a> and the CO2 emissions with <a href="https://www.electricitymap.org/zone/NL">Electricity map</a> Netherlands data (where Google&#39;s European servers are located). </p>
<p>Data shown is for single-GPU training at batch size 60 for a target and memory length of 150, following CMU&#39;s Transformer-XL <a href="https://github.com/kimiyoung/transformer-xl">repo</a>. In order to leverage the Tensor Core capability of the V100, we set batch size 64 and sequence length 152 to use <code>torch.backends.cudnn.benchmark</code> on that GPU. In our model size and speed predictions, we assumed that the inner feed-forward layer dimension was the same as the embedding and attention dimensions, and that the width-to-depth ratio was constant. This is both a good way to save memory, as <em><a href="https://arxiv.org/abs/2001.04451">Reformer</a></em> has shown, and shouldn&#39;t impact performance significantly, as <em><a href="https://arxiv.org/abs/2001.08361">Scaling Laws</a></em> has observed. However, for large scales, we found that the final performance of taller models with a bigger feed-forward layer was consistently better, which is why we give two possible model shapes.</p>
<p><strong>In testing, our predictions always came within 0.06 loss of the actual observed experimental result, very close to the 0.04 run-to-run variance.</strong> If you took the shortcut to the demo before reading, you can come back the start <a href="#start">here</a>!</p>
<h2 id="4-takeaways">4. Takeaways</h2>
<ul>
<li>Big models are surprisingly efficient!</li>
<li>Training until convergence is not efficient at all.</li>
<li>To lower training costs, you should benchmark your model as you do your hyperparameter search to know when to stop.</li>
</ul>

"""