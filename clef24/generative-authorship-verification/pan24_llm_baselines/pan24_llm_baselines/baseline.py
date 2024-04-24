# Copyright 2024 Janek Bevendorff, Webis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import click
from tqdm import tqdm


@click.group()
def main():
    """
    PAN'24 Generative Authorship Detection baselines.
    """
    pass


def comparative_score(score1, score2, epsilon=1e-3):
    """
    Return a single score in [0, 1] based on the comparison of two [0, 1] input scores.

    :param score1: first score
    :param score2: second score
    :param epsilon: non-answer (output score = 0.5) epsilon threshold
    :return: [0, 0.5) if score1 > score2 + eps; (0.5, 1] if score2 > score1 + eps; 0.5 otherwise
    """
    if score1 > score2 + epsilon:
        return min(score2 / 2.0 + 0.5, 1.0)
    if score2 > score1 + epsilon:
        return max((1.0 - score1) / 2.0, 0.0)
    return 0.5


def inverse_comparative_score(score1, score2, epsilon=1e-3):
    return comparative_score(score2, score1, epsilon)


def detect(detector, input_file, output_directory, outfile_name, comp_fn=comparative_score):
    """
    Run a detector on an input file and write results to output directory.

    :param detector: DetectorBase
    :param input_file: input file object
    :param output_directory: output directory path
    :param outfile_name: output filename
    :param comp_fn: function to compare scores
    """
    with open(os.path.join(output_directory, outfile_name), 'w') as out:
        for l in tqdm(input_file, desc='Predicting pairs', unit=' pairs'):
            j = json.loads(l)
            t1 = j['text1']
            t2 = j['text2']

            score1 = detector.get_score(t1)
            score2 = detector.get_score(t2)

            json.dump({'id': j['id'], 'is_human': float(comp_fn(score1, score2))}, out)
            out.write('\n')
            out.flush()


@main.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_directory', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--outfile-name', help='Output file name', default='binoculars.jsonl', show_default=True)
@click.option('-q', '--quantize', type=click.Choice(['4', '8']))
@click.option('-f', '--flash-attn', is_flag=True, help='Use flash-attn 2 (requires Ampere GPU)')
@click.option('--observer', help='Observer model', default='tiiuae/falcon-7b', show_default=True)
@click.option('--performer', help='Performer model', default='tiiuae/falcon-7b-instruct', show_default=True)
@click.option('--device1', help='Observer model device', default='auto', show_default=True)
@click.option('--device2', help='Performer model device', default='auto', show_default=True)
def binoculars(input_file, output_directory, outfile_name, quantize, flash_attn,
               observer, performer, device1, device2):
    """
    PAN'24 baseline: Binoculars.

    References:
    ===========
        Hans, Abhimanyu, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi,
        Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2024.
        “Spotting LLMs with Binoculars: Zero-Shot Detection of Machine-Generated
        Text.” arXiv [Cs.CL]. arXiv. http://arxiv.org/abs/2401.12070.
    """
    from pan24_llm_baselines.detectors import Binoculars

    detector = Binoculars(
        observer_name_or_path=observer,
        performer_name_or_path=performer,
        quantization_bits=quantize,
        use_flash_attn=flash_attn,
        device1=device1,
        device2=device2)
    detect(detector, input_file, output_directory, outfile_name)


@main.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_directory', type=click.Path(file_okay=False, exists=True))
@click.option('-m', '--scoring-mode', type=click.Choice(['lrr', 'npr']), default='lrr', show_default=True)
@click.option('-o', '--outfile-name', help='Output file name', default='detectgpt.jsonl', show_default=True)
@click.option('-s', '--span-length', type=int, default=2, show_default=True, help='Size of mask token spans')
@click.option('-p', '--perturb-pct', type=click.FloatRange(0, 1), default=0.3, show_default=True,
              help='Percentage of tokens to perturb')
@click.option('-n', '--n-samples', type=int, default=20, show_default=True,
              help='Number of perturbed samples to generate for NPR')
@click.option('-b', '--batch-size', type=int, default=5, help='GPU task batch size')
@click.option('-q', '--quantize', type=click.Choice(['4', '8']))
@click.option('-f', '--flash-attn', is_flag=True, help='Use flash-attn 2 (requires Ampere GPU)')
@click.option('--base-model', help='Base detection model', default='tiiuae/falcon-7b', show_default=True)
@click.option('--perturb-model', help='Perturbation model for NPR', default='t5-large', show_default=True)
@click.option('--device1', help='Base model device', default='auto', show_default=True)
@click.option('--device2', help='Perturbation model device', default='auto', show_default=True)
def detectllm(input_file, output_directory, scoring_mode, outfile_name, span_length, perturb_pct, n_samples,
              batch_size, quantize, flash_attn, base_model, perturb_model, device1, device2):
    """
    PAN'24 baseline: DetectLLM.

    References:
    ===========
        Su, Jinyan, Terry Yue Zhuo, Di Wang, and Preslav Nakov. 2023. “DetectLLM: Leveraging
        Log Rank Information for Zero-Shot Detection of Machine-Generated Text.”
        arXiv [Cs.CL]. arXiv. http://arxiv.org/abs/2306.05540.
    """
    from pan24_llm_baselines.detectors import DetectLLM
    from pan24_llm_baselines.perturbators import T5MaskPerturbator

    perturbator = T5MaskPerturbator(
        model_name=perturb_model,
        device=device2,
        span_length=span_length,
        mask_pct=perturb_pct,
        batch_size=batch_size)
    detector = DetectLLM(
        scoring_mode=scoring_mode,
        base_model=base_model,
        quantization_bits=quantize,
        use_flash_attn=flash_attn,
        perturbator=perturbator,
        n_samples=n_samples,
        batch_size=batch_size,
        device=device1)
    detect(detector, input_file, output_directory, outfile_name)


@main.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_directory', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--outfile-name', help='Output file name', default='detectgpt.jsonl', show_default=True)
@click.option('-s', '--span-length', type=int, default=2, show_default=True, help='Size of mask token spans')
@click.option('-p', '--perturb-pct', type=click.FloatRange(0, 1), default=0.3, show_default=True,
              help='Percentage of tokens to perturb')
@click.option('-n', '--n-samples', type=int, default=20, show_default=True,
              help='Number of perturbed samples to generate')
@click.option('-b', '--batch-size', type=int, default=5, help='GPU task batch size')
@click.option('-q', '--quantize', type=click.Choice(['4', '8']))
@click.option('-f', '--flash-attn', is_flag=True, help='Use flash-attn 2 (requires Ampere GPU)')
@click.option('--base-model', help='Base detection model', default='tiiuae/falcon-7b', show_default=True)
@click.option('--perturb-model', help='Perturbation model', default='t5-large', show_default=True)
@click.option('--device1', help='Base model device', default='auto', show_default=True)
@click.option('--device2', help='Perturbation model device', default='auto', show_default=True)
def detectgpt(input_file, output_directory, outfile_name, span_length, perturb_pct, n_samples, batch_size,
              quantize, flash_attn, base_model, perturb_model, device1, device2):
    """
    PAN'24 baseline: DetectGPT.

    References:
    ===========
        Mitchell, Eric, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning,
        and Chelsea Finn. 2023. “DetectGPT: Zero-Shot Machine-Generated Text
        Detection Using Probability Curvature.” arXiv [Cs.CL]. arXiv.
        http://arxiv.org/abs/2301.11305.
    """
    from pan24_llm_baselines.detectors import DetectGPT
    from pan24_llm_baselines.perturbators import T5MaskPerturbator

    perturbator = T5MaskPerturbator(
        model_name=perturb_model,
        device=device2,
        span_length=span_length,
        mask_pct=perturb_pct,
        batch_size=batch_size)
    detector = DetectGPT(
        base_model=base_model,
        quantization_bits=quantize,
        use_flash_attn=flash_attn,
        perturbator=perturbator,
        n_samples=n_samples,
        batch_size=batch_size,
        device=device1)
    detect(detector, input_file, output_directory, outfile_name)


@main.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_directory', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--outfile-name', help='Output file name', default='ppmd.jsonl', show_default=True)
def ppmd(input_file, output_directory, outfile_name):
    """
    PAN'24 baseline: Compression-based cosine.

    References:
    ===========
        Sculley, D., and C. E. Brodley. 2006. “Compression and Machine Learning: A New Perspective
        on Feature Space Vectors.” In Data Compression Conference (DCC’06), 332–41. IEEE.

        Halvani, Oren, Christian Winter, and Lukas Graner. 2017. “On the Usefulness of Compression
        Models for Authorship Verification.” In ACM International Conference Proceeding Series. Vol.
        Part F1305. Association for Computing Machinery. https://doi.org/10.1145/3098954.3104050.
    """

    from pan24_llm_baselines.detectors import PPMdDetector
    detector = PPMdDetector()
    detect(detector, input_file, output_directory, outfile_name)


@main.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_directory', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--outfile-name', help='Output file name', default='length.jsonl', show_default=True)
def length(input_file, output_directory, outfile_name):
    """
    PAN'24 baseline: Text length.
    """
    with open(os.path.join(output_directory, outfile_name), 'w') as out:
        for l in tqdm(input_file, desc='Predicting cases'):
            j = json.loads(l)
            l1 = len(j['text1'])
            l2 = len(j['text2'])
            json.dump({'id': j['id'], 'is_human': float(l1 < l2)}, out)
            out.write('\n')


@main.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_directory', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--outfile-name', help='Output file name', default='unmasking.jsonl', show_default=True)
def unmasking(input_file, output_directory, outfile_name):
    """
    PAN'24 baseline: Authorship unmasking.

    References:
    ===========
        Koppel, Moshe, and Jonathan Schler. 2004. “Authorship Verification as a One-Class
        Classification Problem.” In Proceedings, Twenty-First International Conference on
        Machine Learning, ICML 2004, 489–95.

        Bevendorff, Janek, Benno Stein, Matthias Hagen, and Martin Potthast. 2019. “Generalizing
        Unmasking for Short Texts.” In Proceedings of the 2019 Conference of the North, 654–59.
        Stroudsburg, PA, USA: Association for Computational Linguistics.
    """
    from pan24_llm_baselines.detectors import UnmaskingDetector
    detector = UnmaskingDetector()
    detect(detector, input_file, output_directory, outfile_name)


if __name__ == '__main__':
    main()
