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
using TensorFlow;
using TensorFlow.Island.Classes;

namespace TensorFlow.Island.Tests
{
    public class GraphOperationsTest: Test
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

        public void When_CallingOpAddWithScalars_12_19_Expect_31()
        {
            var lGraph = m_session.Graph;
            var a = lGraph.OpConst(12);
            var b = lGraph.OpConst(19);
            var c = lGraph.OpAdd(a, b);
            Tensor result = m_session.Runner.Run(c);
            var (success, sum) = result.ScalarValueAs<Integer>();
            Assert.IsTrue(success);
            Assert.AreEqual(sum, 12 + 19);
        }
    }
}