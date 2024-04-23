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

from abc import ABC, abstractmethod
from typing import List, Union

__all__ = ['PerturbatorBase']


class PerturbatorBase(ABC):
    """
    LLM detector base class.
    """

    @abstractmethod
    def _perturb_impl(self, text: List[str], n_variants) -> List[str]:
        """
        Perturbation implementation. To be overridden.
        """
        pass

    def perturb(self, text: Union[str, List[str]], n_variants: int = 1) -> Union[str, List[str]]:
        """
        Perturb a given text by changing parts of the input.

        :param text: input text or list of input texts
        :param n_variants: number of perturbation variants
        :return: perturbed text
        """
        return_str = isinstance(text, str)
        text = self._perturb_impl([text] if return_str else text, n_variants)
        return text[0] if return_str else text
