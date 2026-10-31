# Native ensemble implementation notices

`thermal.hpp` implements the BZP and upper-triangular MTTK/RGB cell propagation
and centroid-virial pressure conventions with reference to i-PI's
`ipi/engine/barostats.py`, blob `98cf6e6c38366d9580cb48b69e36e29a48e1bb76`:
https://github.com/i-pi/i-pi/blob/main/ipi/engine/barostats.py

The i-PI project offers the MIT license as an option. That option is used for
this C++ adaptation. Its source carries:

**i-PI Copyright (C) 2014-2015 i-PI developers**

## MIT license

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Original license text:
https://github.com/i-pi/i-pi/blob/main/licenses/license_MIT.txt

No PIGLET parameter dataset is bundled. Matrix provenance must be supplied by
the user; selecting the PIGLET mode does not validate a parameter dataset.
