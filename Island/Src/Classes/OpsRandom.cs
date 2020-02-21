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
        public Output RandomNormal(Shape! shape, double mean = 0, double stddev = 1, int? seed = null, string operName = null)
        {
            using (WithScope (MakeName("RandomNormal", operName)) ){
                var shapeTensor = ConvertShapeToOutput(shape);
                var tmean = Const(mean, "mean");
                var tstddev = Const(stddev, "stddev");
                var (graph, local) = GetRandomSeeds(seed);
                var rnd = RandomStandardNormal(shapeTensor, DataType.Double, graph, local);
                var mul = Mul(rnd, tstddev);
                return Add(mul, tmean);
            }
        }

        public Output RandomUniform(Shape! shape, double minval = 0, double maxval = 1, int? seed = null, string operName = null)
        {
            using (WithScope(MakeName ("random_uniform", operName))) {
                var shapeTensor = ConvertShapeToOutput(shape);
                var minvalTensor = Const(minval, "minval");
                var maxvalTensor = Const(maxval, "maxval");
                var (graph, local) = GetRandomSeeds(seed);
                var rnd = RandomUniform(shapeTensor, DataType.Double, graph, local);
                var mul = Mul(rnd, Sub(maxvalTensor, minvalTensor));
                return Add(mul, minvalTensor);
            }
        }

        public Output RandomUniform(Shape! shape, float minval = 0, float maxval = 1, int? seed = null, string operName = null)
        {
            using (WithScope(MakeName("random_uniform", operName))) {
                var shapeTensor = ConvertShapeToOutput(shape);
                var minvalTensor = Const(minval, "minval");
                var maxvalTensor = Const(maxval, "maxval");
                var (graph, local) = GetRandomSeeds(seed);
                var rnd = RandomUniform(shapeTensor, DataType.Float, graph, local);
                var mul = Mul(rnd, Sub(maxvalTensor, minvalTensor));
                return Add(mul, minvalTensor);
            }
        }
    }
}
