from typing import List, Union

import mygrad
from mygrad.ufuncs._ufunc_creators import MyGradBinaryUfunc, MyGradUnaryUfunc

_public_ufunc_names: List[str] = []
_ufuncs: List[Union[MyGradUnaryUfunc, MyGradBinaryUfunc]] = []

for _name in sorted(
    public_name for public_name in dir(mygrad) if not public_name.startswith("_")
):
    attr = getattr(mygrad, _name)
    if isinstance(attr, (MyGradUnaryUfunc, MyGradBinaryUfunc)):
        _ufuncs.append(attr)
        _public_ufunc_names.append(_name)

public_ufunc_names = tuple(_public_ufunc_names)
ufuncs = tuple(_ufuncs)
