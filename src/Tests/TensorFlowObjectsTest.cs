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
    public class TensorFlowObjectsTest: Test
    {
        // Called before each test method.
        public override void Setup()
        {

        }

        // Called after each test method.
        public override void Teardown()
        {

        }

        // Called before first test method.
        public override void SetupTest()
        {

        }

        // Called after last test method.
        public override void TeardownTest()
        {

        }

        public void When_ReferencingDisposedObject_Expect_ObjectDisposedException()
        {
            Shape shp = {1, 5, 10};
            shp.Dispose();
       	    Assert.Throws(()=>shp.NumDims, typeof(ObjectDisposedException));
        }

        public void When_UsingInvalidShapeDimIndex_Expect_InvalidShapeDimIndexExpection()
        {
            Shape shp = {1, 5, 10};
            Assert.AreEqual(shp.NumDims, 3);
            Assert.AreEqual(shp.Dim[0], 1);
            Assert.AreEqual(shp.Dim[1], 5);
            Assert.AreEqual(shp.Dim[2], 10);
            Assert.Throws(()=>shp.Dim[5], typeof(InvalidShapeDimIndexException));
        }

        public void When_CreatingTensorWith2DArray_Expect_Created()
        {
            Tensor tensor = {{1,2,3},{4,5,6}};
            Assert.AreEqual(tensor.Data.DataType, TF_DataType.TF_INT32);
            Assert.AreEqual(tensor.Data.Shape.NumDims, 2);
            Assert.AreEqual(tensor.Data.Shape.Dim[0], 2);
            Assert.AreEqual(tensor.Data.Shape.Dim[1], 3);
            int[..2, ..3] data;
            memcpy(&data[0,0], tensor.Data.Bytes, tensor.Data.NumBytes);
            Assert.AreEqual(data[0,2], 3);
        }

        public void When_CreatingTensorWithString_Expect_Created()
        {
            Tensor tensor = "MySuperCoolTensorFlowApp";
            var (success, str) = tensor.AsScalar<String>();
            Assert.IsTrue(success);
            Assert.AreEqual(str, "MySuperCoolTensorFlowApp");
        }

        public void When_CreatingTensorWithStrings_Expect_Created()
        {
            Tensor tensor = {"Hello", "World", "TensorFlow"};
            var (success, strs) = tensor.AsArray<String>();
            Assert.IsTrue(success);
            Assert.AreEqual(strs.Length, 3);
            Assert.AreEqual(strs[0], "Hello");
            Assert.AreEqual(strs[1], "World");
            Assert.AreEqual(strs[2], "TensorFlow");
        }

        public void When_GettingTensorInfo_Expect_Success()
        {
            var lSession = new Session(); 
            var lOutput = lSession.Graph.OpConst(1);
            var str = lSession.GetTensorInfo(lOutput);
            Assert.AreEqual(str, "Tensor (\"Const_0: 0\", shape=TensorShape([]), dtype=Int32 )");

            lOutput = lSession.Graph.OpConst({1, 2});
            str = lSession.GetTensorInfo(lOutput);
            Assert.AreEqual(str, "Tensor (\"Const_1: 0\", shape=TensorShape([Dimension(2)]), dtype=Int32 )");   

            lOutput = lSession.Graph.OpConst({{1, 2}});
            str = lSession.GetTensorInfo(lOutput);
            Assert.AreEqual(str, "Tensor (\"Const_2: 0\", shape=TensorShape([Dimension(1), Dimension(2)]), dtype=Int32 )"); 
        }

        public void When_PrintingTensorWithInt32Values_Expect_Success()
        {
            Tensor tensor = {{1,2,3},{4,5,6},{7,8,9}};
            var print_str = tensor.Print(1, 6);
            const string validation_str =
                "[ [     1     2     3]  " + '\n' +
                "  [     4     5     6]  " + '\n' +
                "  [     7     8     9] ]";
            Assert.AreEqual(print_str, validation_str);
        }

        public void When_PrintingTensorWithBoolValues_Expect_Success()
        {
            Tensor tensor = {{true,false,true},{false,false,false},{true,true,true}};
            var print_str = tensor.Print(1, 6);
            const string validation_str =
                "[ [  True False  True]  " + '\n' +
                "  [ False False False]  " + '\n' +
                "  [  True  True  True] ]";
            Assert.AreEqual(print_str, validation_str);
        }
    }
}