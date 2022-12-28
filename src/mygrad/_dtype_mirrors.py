import numpy

__all__ = [
    "bool8",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "intp",
    "uintp",
    "float16",
    "float32",
    "float64",
    "half",
    "single",
    "double",
    "longdouble",
]

try:
    bool8 = numpy.bool8
except AttributeError:  # pragma: no cover
    pass

bool_ = numpy.bool_
int8 = numpy.int8
int16 = numpy.int16
int32 = numpy.int32
int64 = numpy.int64
uint8 = numpy.uint8
uint16 = numpy.uint16
uint32 = numpy.uint32
uint64 = numpy.uint64
intp = numpy.intp
uintp = numpy.uintp
float16 = numpy.float16
float32 = numpy.float32
float64 = numpy.float64
half = numpy.half
single = numpy.single
double = numpy.double
longdouble = numpy.longdouble
