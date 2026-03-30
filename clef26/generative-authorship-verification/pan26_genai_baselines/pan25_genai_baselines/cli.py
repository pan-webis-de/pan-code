# Copyright 2025 Janek Bevendorff, Webis
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
    PAN'25 Generative AI Authorship Verification baselines.
    """
    pass


def detect(detector, input_file, output_directory, outfile_name, c_at_1_threshold=0.01):
    """
    Run a detector on an input file and write results to output directory.

    :param detector: DetectorBase
    :param input_file: input file object
    :param output_directory: output directory path
    :param outfile_name: output filename
    :param c_at_1_threshold: c@1 optimization threshold
    """
    with open(os.path.join(output_directory, outfile_name), 'w') as out:
        for l in tqdm(input_file, desc='Predicting texts', unit=' texts'):
            j = json.loads(l)
            score = detector.get_score(j['text'], normalize=True)
            if abs(score - .5) < c_at_1_threshold:
                # Optimize c@1
                score = 0.5
            json.dump({'id': j['id'], 'label': float(score)}, out)
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
@click.option('--device', help='GPU device', default='auto', show_default=True)
def binoculars(input_file, output_directory, outfile_name, quantize, flash_attn,
               observer, performer, device):
    """
    PAN'25 baseline: Binoculars.

    References:
    ===========
        Hans, Abhimanyu, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi,
        Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2024.
        “Spotting LLMs with Binoculars: Zero-Shot Detection of Machine-Generated
        Text.” arXiv [Cs.CL]. arXiv. http://arxiv.org/abs/2401.12070.
    """
    from pan25_genai_baselines.binoculars import Binoculars

    detector = Binoculars(
        observer_name_or_path=observer,
        performer_name_or_path=performer,
        quantization_bits=quantize,
        flash_attn=flash_attn,
        device=device)
    detect(detector, input_file, output_directory, outfile_name)


@main.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_directory', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--outfile-name', help='Output file name', default='ppmd.jsonl', show_default=True)
def ppmd(input_file, output_directory, outfile_name):
    """
    PAN'25 baseline: Compression-based cosine.

    References:
    ===========
        Sculley, D., and C. E. Brodley. 2006. “Compression and Machine Learning: A New Perspective
        on Feature Space Vectors.” In Data Compression Conference (DCC’06), 332–41. IEEE.

        Halvani, Oren, Christian Winter, and Lukas Graner. 2017. “On the Usefulness of Compression
        Models for Authorship Verification.” In ACM International Conference Proceeding Series. Vol.
        Part F1305. Association for Computing Machinery. https://doi.org/10.1145/3098954.3104050.
    """

    from pan25_genai_baselines.ppmd import PPMdDetector
    detector = PPMdDetector()
    detect(detector, input_file, output_directory, outfile_name)


@main.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_directory', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--outfile-name', help='Output file name', default='tfidf.jsonl', show_default=True)
def tfidf(input_file, output_directory, outfile_name):
    """
    PAN'25 baseline: TF-IDF SVM.
    """

    from pan25_genai_baselines.tfidf import TfidfDetector
    detector = TfidfDetector()
    detect(detector, input_file, output_directory, outfile_name, c_at_1_threshold=0.05)


if __name__ == '__main__':
    main()
