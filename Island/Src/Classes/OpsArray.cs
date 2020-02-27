/*
MIT License

Copyright (c) 2019-2020 Wuping Xin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

using RemObjects.Elements.System;
using TensorFlow.Island.Api;

namespace TensorFlow.Island.Classes
{
    public partial class Graph
    {
        public Output Zeros(Shape! shape, DataType dtype = DataType.Double, string opName = null)
        {
            return Constant(0, shape, dtype, opName);
        }

        public Output Ones(Shape! shape, DataType dtype = DataType.Double, string opName = null)
        {
            return Constant(1, shape, dtype, opName);
        }

        // Creates a constant tensor from a tensor-like object. value is a scalar constant.
        // If value's type is different from dtype, it will be cast to dtype.
        public Output Constant(Object value, Shape! shape, DataType dtype = DataType.Double, string opName = null)
        {
            var tensor = Tensor.ObjectToTensor(value, shape, dtype);
            return Const(tensor, opName);
        }

    }
}
