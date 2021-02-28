from typing import List, Type

import mygrad

_public_ufunc_names: List[str] = []
_ufuncs: List[Type[mygrad.ufunc]] = []

for _name in sorted(
    public_name for public_name in dir(mygrad) if not public_name.startswith("_")
):
    attr = getattr(mygrad, _name)
    try:
        if (
            issubclass(attr, mygrad.ufunc) and attr is not mygrad.ufunc
        ):  # abc's are their own subclass...
            _ufuncs.append(attr)
            _public_ufunc_names.append(_name)
    except TypeError:
        continue

public_ufunc_names = tuple(_public_ufunc_names)
ufuncs = tuple(_ufuncs)
