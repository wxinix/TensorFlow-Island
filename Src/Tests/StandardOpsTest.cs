// MIT License
// Copyright (c) 2019-2021 Wuping Xin.
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

        public void Add_Scalars()
        {
            var g = m_session.Graph;
            var x = g.Const(12);
            var y = g.Const(19);
            var addOutput = g.Add(x, y);
            Tensor addResult = m_session.Runner.Run(addOutput);
            var (success, sum) = addResult.AsScalar<Integer>();
            Assert.IsTrue(success);
            Assert.AreEqual(sum, 12 + 19);
        }

        public void Fill_1X2_Vector_With_Value_9()
        {
            var g = m_session.Graph;
            var dims = g.Const({2,3});
            var value = g.Const(9);
            var fillOutput = g.Fill(dims, value);
            Tensor fillResult = m_session.Runner.Run(fillOutput);
            Assert.AreEqual(fillResult.Data.Shape.NumDims, 2);
            Assert.AreEqual(fillResult.Data.Shape.Dim[0], 2);
            Assert.AreEqual(fillResult.Data.Shape.Dim[1], 3);
            Assert.AreEqual(fillResult.Data.NumBytes, 2 * 3 * sizeOf(Integer));
            // With explicit initial values, data is int[2,3] static array;
            // Without explicit initial values, data is dynamic array int[2][3]
            var data = new int[2,3] {{0,0,0},{0,0,0}};  
            // If data is static array, we can do a continuous mem copy below.
            // If dynamic array, we can NOT do continuous mem copy.
            memcpy(&data[0,0], fillResult.Data.Bytes, fillResult.Data.NumBytes);
            Assert.AreEqual(data[1, 2], 9);
        }

        public void Negate_1X3_Vector()
        {
            var g = m_session.Graph;
            var x = g.Const({1, 2, 3});
            var negOutput = g.Neg(x);
            Tensor negResult = m_session.Runner.Run(negOutput);
            var (success, values) = negResult.AsArray<int>();
            Assert.IsTrue(success);
            Assert.AreEqual(values[0], -1);
            Assert.AreEqual(values[1], -2);
            Assert.AreEqual(values[2], -3);
        }

        public void Inv_1X3_Vector()
        {
            var g = m_session.Graph;
            var x = g.Const({1.0, 2.0, 3.0}); // will default to double type.
            var invOutput = g.Inv(x);
            Tensor tensor_inv = m_session.Runner.Run(invOutput);
            var (success, values) = tensor_inv.AsArray<double>();
            Assert.IsTrue(success);
            Assert.AreEqual(values.Length, 3);
            Assert.AreEqual(values[0], 1.0);
            Assert.AreEqual(values[1], 1.0/2.0);
            Assert.AreEqual(values[2], 1.0/3.0);
        }
    }
}