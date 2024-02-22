# Copyright 2020 InterDigital Communications, Inc.
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


from compressai.models import WACNN2,SymmetricalTransFormer7,SymmetricalTransFormer6,conditionalZigzag,SymmetricalTransFormer5,\
    SymmetricalTransFormer4,SymmetricalTransFormer3,SymmetricalTransFormer2,ClipEncoder4,\
    ClipEncoder3, ClipEncoder, SymmetricalTransFormer, WACNN,ConditionalResidualCoding

from .pretrained import load_pretrained as load_state_dict

models = {
    'stf2': ClipEncoder,
    'stf3': ClipEncoder3,
	'stf4': ClipEncoder4,
    'stf': SymmetricalTransFormer,
    'stf5': SymmetricalTransFormer2,
    'stf6': SymmetricalTransFormer3,
    'stf7': SymmetricalTransFormer4,
    'stf8': SymmetricalTransFormer5,
    'stf9': SymmetricalTransFormer6,
    'stf10': SymmetricalTransFormer7,
    'stf11': ConditionalResidualCoding,
    'czigzag': conditionalZigzag, 
    'cnn': WACNN,
    'cnn2': WACNN2,
}
