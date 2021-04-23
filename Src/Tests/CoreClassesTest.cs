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
using RemObjects.Elements.RTL;
using RemObjects.Elements.System;
using TensorFlow.Island.Classes;

namespace TensorFlow.Island.Tests
{
    public class CoreClassesTest: Test
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

        public void Will_Raise_ObjectDisposedException_On_Disposed_Object()
        {
            Shape shp = {1, 5, 10};
            shp.Dispose();
       	    Assert.Throws(()=>shp.NumDims, typeof(ObjectDisposedException));
        }

        public void Will_Raise_ArgumentOutOfRangeException_On_Invalid_ShapeDim()
        {
            Shape shp = {1, 5, 10};
            Assert.AreEqual(shp.NumDims, 3);
            Assert.AreEqual(shp.Dim[0], 1);
            Assert.AreEqual(shp.Dim[1], 5);
            Assert.AreEqual(shp.Dim[2], 10);
            Assert.Throws(()=>shp.Dim[5], typeof(ArgumentOutOfRangeException));
        }

        public void Can_Create_Tensor_With_2DArray()
        {
            Tensor tensor = {{1,2,3},{4,5,6}};
            Assert.AreEqual(tensor.Data.Type, DataType.Int32);
            Assert.AreEqual(tensor.Data.Shape.NumDims, 2);
            Assert.AreEqual(tensor.Data.Shape.Dim[0], 2);
            Assert.AreEqual(tensor.Data.Shape.Dim[1], 3);
            int[..2, ..3] data;
            memcpy(&data[0,0], tensor.Data.Bytes, tensor.Data.NumBytes);
            Assert.AreEqual(data[0,2], 3);
        }

        public void Can_Create_Tensor_With_String()
        {
            Tensor tensor = "MySuperCoolTensorFlowApp";
    		(bool success, string str) = tensor.AsScalar<String>();
            Assert.IsTrue(success);
            Assert.AreEqual(str, "MySuperCoolTensorFlowApp");
        }

        public void Can_Create_Tensor_With_Strings()
        {
            Tensor tensor = {"Hello", "World", "TensorFlow"};
            var (success, strs) = tensor.AsArray<String>();
            Assert.IsTrue(success);
            Assert.AreEqual(strs.Length, 3);
            Assert.AreEqual(strs[0], "Hello");
            Assert.AreEqual(strs[1], "World");
            Assert.AreEqual(strs[2], "TensorFlow");
        }

        public void Can_Get_Tensor_Info()
        {
            var session = new Session(); 
            var output = session.Graph.Const(1L);
            var str = session.GetTensorInfo(output);
            Assert.AreEqual(str, "Tensor (\"Const_0: 0\", shape=TensorShape([]), dtype=Int64)");

            output = session.Graph.Const({1, 2});
            str = session.GetTensorInfo(output);
            Assert.AreEqual(str, "Tensor (\"Const_1: 0\", shape=TensorShape([Dimension(2)]), dtype=Int32)");   

            output = session.Graph.Const({{1, 2}});
            str = session.GetTensorInfo(output);
            Assert.AreEqual(str, "Tensor (\"Const_2: 0\", shape=TensorShape([Dimension(1), Dimension(2)]), dtype=Int32)"); 
       }

        public void Can_Print_Tensor_With_Int32_Values()
        {
            Tensor tensor = {{1,2,3},{4,5,6},{7,8,9}};
            var printStr = tensor.Print(aMaxBytesAllowed: 1000) DecimalDigits(1) MaxWidth(6);
            const string validationStr =
                "[ [     1     2     3]  " + '\n' +
                "  [     4     5     6]  " + '\n' +
                "  [     7     8     9] ]";
            Assert.AreEqual(printStr, validationStr);
        }

        public void Can_Print_Tensor_With_Bool_Values()
        { 
			Tensor tensor = {{true,false,true},{false,false,false},{true,true,true}};
			var printStr = tensor.Print(aMaxBytesAllowed: 1000) DecimalDigits(1) MaxWidth(6);
			const string validationStr =
				"[ [  True False  True]  " + '\n' +
				"  [ False False False]  " + '\n' +
				"  [  True  True  True] ]";
			Assert.AreEqual(printStr, validationStr);            
        }
    } 
}