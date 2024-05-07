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
import hashlib
import os
from typing import Iterable, List, Optional, Union

from more_itertools import chunked

__all__ = ['PerturbatorBase']


class PerturbatorBase:
    """
    LLM detector base class.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        :param cache_dir: cache directory for storing generated perturbations
        """
        self._cache_dir = cache_dir

    @property
    def cache_dir(self) -> Optional[str]:
        """
        Base directory for caching perturbations.
        """
        if self._cache_dir and (k := self.cache_key):
            return os.path.join(self._cache_dir, k)
        return self._cache_dir

    @property
    def cache_key(self) -> Optional[str]:
        """
        Optional cache key.
        Should be unique for each set of hyperparameters. Must be a valid directory name.
        """
        return None

    def _perturb_impl(self, text: List[str], n_variants: int) -> Iterable[str]:
        """
        Perturbation implementation. To be overridden.

        :param text: list of input texts
        :param n_variants: number of variants to generate
        :return: iterable or generator of perturbed texts
        """
        raise NotImplementedError

    def _generate_and_cache(self, text: List[str], n_variants: int):
        pert = self._perturb_impl(text, n_variants)
        for orig_t, chunk in zip(text, chunked(pert, n_variants)):
            h = hashlib.sha256(orig_t.encode(errors='ignore')).hexdigest()
            with open(os.path.join(self.cache_dir, h), 'a') as f:
                f.write('\n'.join(json.dumps(ci, ensure_ascii=False) for ci in chunk))
                f.write('\n')
        return pert

    def _get_cached(self, text: List[str], n_variants: int):
        if not self.cache_dir:
            yield from self._perturb_impl(text, n_variants)
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        uncached_wait_list = []
        for t in text:
            h = hashlib.sha256(t.encode(errors='ignore')).hexdigest()
            cache_name = os.path.join(self.cache_dir, h)

            if not os.path.isfile(cache_name):
                # No cache file found, just add to list of variants to be generated later
                uncached_wait_list.append(t)
                continue

            # We found a cache file, first execute and clear out wait list of uncached variants
            if uncached_wait_list:
                yield from self._generate_and_cache(uncached_wait_list, n_variants)
                uncached_wait_list = []

            # Then read cache file and add to list of variants
            with open(cache_name, 'r') as f_:
                variants = [json.loads(l) for i, l in enumerate(f_) if i < n_variants]

            # Generate and cache more variants if needed
            if len(variants) < n_variants:
                variants.extend(self._generate_and_cache([t], n_variants - len(variants)))

            yield from variants

        # Generate and yield any remaining previously uncached variants
        if uncached_wait_list:
            yield from self._generate_and_cache(uncached_wait_list, n_variants)

    def perturb(self, text: Union[str, List[str]], n_variants: int = 1) -> Union[str, List[str]]:
        """
        Perturb a given text by changing parts of the input.

        :param text: input text or list of input texts
        :param n_variants: number of perturbation variants
        :return: perturbed text
        """
        if isinstance(text, str):
            return next(self._get_cached([text], n_variants))
        return list(self._get_cached(text, n_variants))
