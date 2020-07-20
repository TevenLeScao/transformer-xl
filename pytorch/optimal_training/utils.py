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


def co2_to_trees(co2):
    return co2 / 60 * 3650


def co2_to_kms(co2):
    return co2 / 0.403 * 1.60934


def energy_fill(kWh, co2):
    return 'This will consume about <span style="font-weight: bold">{:.2f}</span> ' \
           'kWh, releasing <span style="font-weight: bold">{:.2f}</span> ' \
           'kgs of CO2. That is equivalent to <span style="font-weight: bold">{:.2f}</span> ' \
           'kms with an average American passenger car and could be offset ' \
           'by growing a tree for <span style="font-weight: bold">{:.2f}</span> ' \
           'days.<sup><a href=' \
           '"https://www.epa.gov/energy/greenhouse-gases-equivalencies-calculator-calculations-and-references#miles"' \
           '>1</a></sup>'.format(kWh, co2, co2_to_kms(co2), co2_to_trees(co2))


md1 = """<h1 id="how-big-should-my-language-model-be">How Big Should My Language Model Be?</h1>
<img class='center' style='height: 5em; float: right;' src='https://raw.githubusercontent.com/TevenLeScao/transformer-xl/master/pytorch/assets/avatar_logo_joint.png' alt='avatar'>
<h4>Published on June 08, 2020.</h4> 
<h4>Teven Le Scao, researcher at Hugging Face • <a href="https://twitter.com/Fluke_Ellington">@Fluke_Ellington</a> </h4> 
<p>Natural Language Processing can sometimes feel like model size is optimized for headlines. <a href="https://arxiv.org/abs/2005.14165">175 billion parameters</a> is certainly an eye-catching number! Why not just train more efficiently with a smaller model? One surprising scaling effect of deep learning is that <strong>bigger neural networks are actually compute-efficient.</strong> This is something OpenAI in particular has explored in papers like <em><a href="https://arxiv.org/abs/2001.08361">Scaling Laws for Neural Language Models</a></em>. Research at Hugging Face also leverages this phenomenon, and we&#39;ve combined it with GPU speed estimations to ensure model size is just right for the compute budget of the experiment (when in doubt, it&#39;s bigger than you think!). This blog post will show how this impacts architecture decisions on a standard language modeling benchmark: <strong>we replicate the 14-layer state-of-the-art result from <a href="https://arxiv.org/pdf/1901.02860.pdf">Zhang et al.&#39;s Transformer-XL paper</a> without any hyper-parameter optimization and saving 25% of training time</strong>. We also estimate that <strong>the 18-layer model from the same paper trained for an order of magnitude too many training steps.</strong> <a name="start">Wanna</a> play with our demo before reading? Just click <a href="#demo">here</a>!</p>
<h2 id="1-there-is-an-optimal-time-to-stop-training-and-its-earlier-than-you-think">1. There is an optimal time to stop training (and it&#39;s earlier than you think)</h2>
<p>Let&#39;s look at some loss curves. For our example, the task will be training Transformer-XL, the state-of-the-art in language modeling, on Wikitext-103, a standard, medium-size benchmark. GPT-2 doesn&#39;t perform well on this dataset scale. As training progresses, we&#39;ll look at the performance of the model (as measured by validation loss) depending on compute cost (as measured by floating point operations). Let&#39;s run a few experiments! In the following plot, every line of colour corresponds to a Transformer-XL run of 200000 steps with a different number and size of layers, with all other hyperparameters kept the same. This spans models from a mere thousand to a hundred million parameters (excluding embeddings). Bigger models are on the right as they require more compute for every step. Don&#39;t worry, we&#39;ve already run them so you don&#39;t have to. All graphs are interactive, play with them!</p>"""
md2 = """
<p>As introduced in <em>Scaling Laws</em>, we plot the validation loss against non-embedding floating-point operations (neFLOs). There seems to be a frontier of performance for a given neFLO budget that no model manages to beat, depicted here in red. In <em>Scaling Laws</em>, it is referred to as the compute frontier. Every run reaches it, or comes close, after an initial phase of quick loss improvement, then tapers away as the end of training is not as efficient. This has a very practical implication: if you have a certain budget in floating-point operations, to reach the best performance, you should choose a model size that reaches the compute frontier after that many operations and stop it at that moment. This is actually way before model convergence, which usually happens around 10 times later! In essence, if you have extra compute budget, you should invest most of it in a bigger model, and only a small part in more training steps. In <em>Scaling Laws</em>, the OpenAI team fitted a power law to the compute frontier on GPT-2 training. This still seems to be a good fit in our task. In addition, we also fitted a power law between the compute budget and the number of parameters of the model that is optimal for that budget. It is pictured in the following plot.</p>

"""
md3 = """
<p>As good models tend to spend considerable time tangent on the compute frontier, there is a bit of noise in the relationship. However, this also means that there is more tolerance in the estimation even if the model size we predict is a bit off, as the imperfect model will still be very close to optimal. We find that <strong>if the compute budget is multiplied by 10, the optimal model size is multiplied by 7.41 and the number of optimal training steps by only 1.35</strong>. Extrapolating with this rule to the much-bigger 18-layer SOTA model from Zhang et al., we find that <strong>its optimal number of training steps was around 250000</strong>. Even if this number is imprecise due to the change of scale, <strong>it is much smaller than the 4 million steps from their replication script</strong>. Starting from an even bigger model and stopping earlier would have yielded a better loss for that (huge) compute budget.</p>
<h2 id="2-gpus-are-optimized-for-large-wide-models">2. GPUs are optimized for large, wide models</h2>
<p>We now have a rule connecting performance and optimal size with neFLOs. However, neFLOs are a bit hard to picture. Can we translate that into a more immediate resource, like training time? Whether you are constrained by temporal or financial constraints, the main resource is GPU time. In order to establish a connection between neFLOs and GPU time, we benchmarked different Transformer-XL model sizes on 4 different GPUs available on Google Cloud Platform across tens of thousands of runs, taking into account mixed precision training. Here are our findings:</p>
<h5 id="speed-estimation">Speed estimation</h5>
<p>neFLOs per second speed can be modeled as a factorized multivariate function (sounds scary, but this just means the equation can be written simply as below) of model width (the number of neurons per layer), depth (the number of layers) and batch size, by increasing order of importance. In our estimations, the maximum prediction error was 15% of the observed speed.</p>
<img class='center' style='height: 1.25em;' src='https://raw.githubusercontent.com/TevenLeScao/transformer-xl/master/pytorch/assets/formula_1.png' alt='formula_1'>
<h5 id="width">Width</h5>
<p>GPUs are optimized for the large feed-forward layers of wide transformers. In all of our experiments, neFLOs per second depended on model width as <strong>a power law of exponent around 1.6</strong>. This means that a model that&#39;s twice as wide, which requires 4 times more operations, also goes through those operations around 3.16 times faster, <strong>nearly offsetting the additional compute cost</strong>.</p>
<h5 id="depth">Depth</h5>
<p>neFLOs per second were also positively correlated with depth. Our best results were attained by modeling this connection as proportional to depth * (depth + additive constant). This is coherent with the fact that Transformers must process layers serially. In essence, <strong>deeper models aren&#39;t actually faster, but they appear to be so as their overhead is smaller relative to the more productive operations</strong>. The additive constant, which represents this overhead, was consistently around 5 in our experiments, which essentially means that data loading to the GPU, embeddings, and softmax operations, represent around 5 transformer layers&#39; worth of time.</p>
<h5 id="batch-size">Batch size</h5>
<p>Batch size played the least role. It was <strong>positively correlated with speed for small values, but quickly saturated</strong> (and even seemed to hurt at high values, after 64 on the V100 and P100 and 16 on the K80 and P4). We modeled its contribution as a logarithmic function to keep things simple as it was also the variable for which the factorized independence assumption was the weakest. We ran all our experiments at size 64 on a single GPU. This is another perk of big models: <strong>as bigger batch sizes don&#39;t seem to help much, if your model is too big to fit on a GPU, you could just use a smaller batch size and gradient accumulation.</strong></p>
<h5 id="powers-of-2-still-matter-in-2020">Powers of 2 still matter in 2020!</h5>
<p>Finally, one surprising takeaway was that <strong>hyperparameters whose width or batch size were powers of 2 out-performed the others</strong>. That was the case on GPUs with and without Tensor Core capability. On Tensor Core GPUs like the V100, NVIDIA recommends tensor shapes that are multiples of 8; however, we kept seeing improvements beyond that, up to multiples of 512. In the end, we only fitted on powers of 2 as fitting on all data points meant a poor fit quality that consistently under-estimated speed for powers of 2 points, and one might as well choose the fastest parameters.</p>
<p>In the end, our final estimation of operation speed was as follows:</p>
<img class='center' style='height: 2.5em;' src='https://raw.githubusercontent.com/TevenLeScao/transformer-xl/master/pytorch/assets/formula_2.png' alt='formula_2'>
<p>with, for example on a V100 GPU without mixed precision, k=2.21*10<sup>7</sup>, a=1.66, b=5.92, and c=1.33. Different GPUs had close results with a different multiplicative constant.</p>
<h2 id="3-demonstration-on-a-language-modeling-task-wikitext-103">3. <a name="demo">Demonstration on a language modeling task: Wikitext-103</a></h2>
<p>Now that we have obtained a relation between model size and training speed, we can predict, for a certain GPU time or price budget, the optimal model size on the task and the performance it will achieve.</p>

"""
md4 = """<p>Prices are indicated for Google Cloud Platform. The energy consumption was estimated thanks to Peter Henderson&#39;s <a href="https://github.com/Breakend/experiment-impact-tracker">Experiment impact tracker</a> and the CO2 emissions with <a href="https://www.electricitymap.org/zone/NL">Electricity map</a> Netherlands data (where Google&#39;s European servers are located). Even though huge training costs make headlines, it is still possible to replicate a state-of-the-art result on a medium-size dataset for thirty bucks! A single V100 with properly optimized training is already quite a powerful weapon.</p>
<p>Data shown is for single-GPU training at batch size 60 on Wikitext-103 for a target and memory length of 150, following CMU&#39;s Transformer-XL <a href="https://github.com/kimiyoung/transformer-xl">repo</a>. In order to leverage the Tensor Core capability of the V100, we set batch size 64 and sequence length 152 on that GPU. In our model size and speed predictions, we assumed that the inner feed-forward layer dimension was the same as the embedding and attention dimensions, and that the width-to-depth ratio was constant. This is a good way to save memory, as <em><a href="https://arxiv.org/abs/2001.04451">Reformer</a></em> has shown. <em><a href="https://arxiv.org/abs/2001.08361">Scaling Laws</a></em> has observed that shape doesn&#39;t impact performance significantly in GPT-2. However, for large scales, we found that the final performance of taller models with a bigger feed-forward layer was consistently better, which is why we give two possible model shapes. </p>
<p>In order to replicate the result of the medium-size Transformer-XL pre-trained model (3.15 loss), we tweaked our example model size to add a bigger feed-forward dimension and have high powers of 2 while keeping the same number of parameters. This gave us a model of 14 layers with 768 hidden dimensions and 1024 feed-forward dimensions. In comparison, the CMU pre-trained model was found through aggressive hyper-parameter search with a much more unusual shape of 16 layers of 410 hidden dimensions and 2100 feed-forward dimensions. In our experiment, even though it was 50% bigger, our model was actually 20% faster per batch on an NVIDIA RTX Titan as its shapes were high powers of 2, and it was a shorter, wider model. For that model, the script provided by the CMU team was already very close to optimal stopping time; in the end, we obtained the same performance with <strong>25% less training time</strong>. Most importantly, this was the case even though the pre-trained model&#39;s hyper-parameter tuning gave it a much more optimized shape, and we had also kept the same random seed it was tuned with. Since we calculated our scaling laws with much smaller-scale trainings, saving on parameter search might actually be the bigger gain here. If you took the shortcut to the demo before reading, you can come back the start <a href="#start">here</a>!</p>
<h2 id="4-takeaways">4. Takeaways</h2>
<ul>
<li>Big models are surprisingly efficient!</li>
<li>Training until convergence is not efficient at all.</li>
<li>Benchmarking smaller-scale runs allows us to predict model performance and optimal stopping time for production-scale models.</li>
<li>Using larger models stopped earlier and optimizing model size for speed lowers training costs.</li>
</ul>
<p>I built this tool automatically using the data from our Transformer-XL runs. If you are interested in having this feature available for other NLP tasks as part of the Hugging Face repository, you can contact me on Twitter at <a href="https://twitter.com/Fluke_Ellington">@Fluke_Ellington</a>, drop me a mail at <code>teven@huggingface.co</code>, or add a reaction on <a href="https://github.com/huggingface/transformers/issues/4847">our Github issue</a>!</p>

"""