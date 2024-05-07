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

import click
from tqdm import tqdm


@click.group()
def main():
    """
    Generate and cache text perturbations for DetectGPT/DetectLLM.
    """


@main.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.option('--perturb-model', help='Perturbation model', default='t5-3b', show_default=True)
@click.option('--device', help='Perturb model device', default='auto', show_default=True)
@click.option('-s', '--span-length', type=int, default=2, show_default=True, help='Size of mask token spans')
@click.option('-p', '--perturb-pct', type=click.FloatRange(0, 1), default=0.3, show_default=True,
              help='Percentage of tokens to perturb')
@click.option('-n', '--n-samples', type=int, default=20, show_default=True,
              help='Number of perturbed samples to generate')
@click.option('-b', '--batch-size', type=int, default=20, help='GPU task batch size')
@click.option('-q', '--quantize', type=click.Choice(['4', '8']))
@click.option('-f', '--flash-attn', is_flag=True, help='Use flash-attn 2 (requires Ampere GPU)')
def t5(input_file, output_dir, perturb_model, device, span_length, perturb_pct,
       n_samples, batch_size, quantize, flash_attn):
    """
    Generate and cache T5 mask perturbations.
    """

    from pan24_llm_baselines.perturbators import T5MaskPerturbator
    pert = T5MaskPerturbator(
        cache_dir=output_dir,
        model_name=perturb_model,
        quantization_bits=quantize,
        use_flash_attn=flash_attn,
        device=device,
        span_length=span_length,
        mask_pct=perturb_pct,
        batch_size=batch_size)

    t = set()
    for l in tqdm(input_file, desc='Loading inputs', unit=' files'):
        j = json.loads(l)
        if 'text' in j:
            t.add(j['text'])
        elif 'text1' in j and 'text2' in j:
            t.add(j['text1'])
            t.add(j['text2'])
        else:
            raise click.UsageError('Invalid input schema. Expected either "text" or "text1" / "text2" keys.')

    pert.perturb(list(t), n_samples)


if __name__ == '__main__':
    main()
