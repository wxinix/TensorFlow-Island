// MIT License
// Copyright (c) 2019-2020 Wuping Xin.
//
// Permission is hereby  granted, free of charge, to any  person obtaining a copy
// of this software and associated  documentation files (the "Software"), to deal
// in the Software  without restriction, including without  limitation the rights
// to  use, copy,  modify, merge,  publish, distribute,  sublicense, and/or  sell
// copies  of  the Software,  and  to  permit persons  to  whom  the Software  is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
// IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
// FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT  SHALL THE
// AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM,  DAMAGES OR  OTHER
// LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

using RemObjects.Elements.EUnit;
using TensorFlow.Island.Classes;

namespace TensorFlow.Island.Tests
{
    public class StandardOpsTest: Test
    {
        private Session m_session;
    
        // Called before each test method.
        public override void Setup()
        {
            m_session = new Session();
        }

        // Called after each test method.
        public override void Teardown()
        {
            m_session.Dispose();
        }

        // Called before first test method.
        public override void SetupTest()
        {

        }

        // Called after last test method.
        public override void TeardownTest()
        {

        }

        public void Add_X_As_12_And_Y_As_19_Calculated()
        {
            var lGraph = m_session.Graph;
            var input_x = lGraph.Const(12);
            var input_y = lGraph.Const(19);
            var output_sum = lGraph.Add(input_x, input_y);
            Tensor tensor_sum = m_session.Runner.Run(output_sum);
            var (success, sum_value) = tensor_sum.AsScalar<Integer>();
            Assert.IsTrue(success);
            Assert.AreEqual(sum_value, 12 + 19);
        }

        public void Fill_2X3_Vector_With_Value_9_Calculated()
        {
            var lGraph = m_session.Graph;
            var output_fill = lGraph.Fill(lGraph.Const({2,3}), lGraph.Const(9));
            Tensor tensor_fill = m_session.Runner.Run(output_fill);
            Assert.AreEqual(tensor_fill.Data.Shape.NumDims, 2);
            Assert.AreEqual(tensor_fill.Data.Shape.Dim[0], 2);
            Assert.AreEqual(tensor_fill.Data.Shape.Dim[1], 3);
            Assert.AreEqual(tensor_fill.Data.NumBytes, 2 * 3 * sizeOf(Integer));
            // With explicit initial values, data is int[2,3] static array;
            // Without explicit initial values, data is dynamic array int[2][3]
            var data = new int[2,3] {{0,0,0},{0,0,0}};  
            // If data is static array, we can do a continuous mem copy below.
            // If dynamic array, we can NOT do continuous mem copy.
            memcpy(&data[0,0], tensor_fill.Data.Bytes, tensor_fill.Data.NumBytes);
            Assert.AreEqual(data[1, 2], 9);
        }

        public void Negate_X_As_1X3_Vector_Caculated()
        {
            var lGraph = m_session.Graph;
            var input_x = lGraph.Const({1, 2, 3});
            var output_neg = lGraph.Neg(input_x);
            Tensor tensor_neg = m_session.Runner.Run(output_neg);
            var (success, values) = tensor_neg.AsArray<int>();
            Assert.IsTrue(success);
            Assert.AreEqual(values[0], -1);
            Assert.AreEqual(values[1], -2);
            Assert.AreEqual(values[2], -3);
        }

        public void Inv_X_As_1X3_Vector_Caculated()
        {
            var lGraph = m_session.Graph;
            var input_x = lGraph.Const({1.0, 2.0, 3.0}); // will default to double type.
            var output_inv = lGraph.Inv(input_x);
            Tensor tensor_inv = m_session.Runner.Run(output_inv);
            var (success, values) = tensor_inv.AsArray<double>();
            Assert.IsTrue(success);
            Assert.AreEqual(values.Length, 3);
        }
    }
}