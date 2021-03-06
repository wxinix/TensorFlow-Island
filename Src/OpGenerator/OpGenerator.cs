﻿/*
MIT License Copyright(c) 2019-2021 Wuping Xin.

Permission is hereby granted, free of charge, to  any  person obtaining a copy
of this software and associated documentation files (the "Software"), to  deal
in the Software  without restriction,  including without limitation the rights
to use, copy, modify,  merge,  publish,  distribute, sublicense,  and/or  sell
copies   of   the  Software, and  to permit persons to  whom  the Software  is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
IMPLIED, INCLUDING BUT  NOT  LIMITED TO THE   WARRANTIES OF  MERCHANTABILITY,
FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO  EVENT SHALL THE
AUTHORS  OR COPYRIGHT  HOLDERS BE  LIABLE  FOR  ANY CLAIM,  DAMAGES OR   OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING  FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This is the driver for the operation generator, using the information provided
by the Tensorflow run-time to produce strongly-typed and high level methods on
the TensorFlow.Island.Graph class.  The  output is a  partial class that is in
line with TensorFlow.Island library.

Originally Developed by:
   Miguel de Icaza, Author of TensorFlowSharp, Copyright 2017, Microsoft.

Adapted for TensorFlow.Island by:
   Wuping Xin, Copyright 2020, KLD Engineering, P. C.

 */

// Warns when a culture-aware 'StartsWith' call is used by default.
#pragma warning disable RECS0063

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using ProtoBuf;
using Tensorflow;

namespace TensorFlow.Island.OpGenerator
{
    [StructLayout(LayoutKind.Sequential)]
    struct Buffer
    {
        public IntPtr data;
        public UInt64 length;
        public IntPtr data_deallocator;
    }

    class Status : IDisposable
    {
        private IntPtr _handle;

        [DllImport("tensorflow")]
        private static extern unsafe IntPtr TF_NewStatus();
        [DllImport("tensorflow")]
        private static extern unsafe void TF_DeleteStatus(IntPtr status);
        [DllImport("tensorflow")]
        private static extern unsafe int TF_GetCode(IntPtr status);

        public Status() => _handle = TF_NewStatus();
        ~Status() => Dispose();

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                TF_DeleteStatus(_handle);
                _handle = IntPtr.Zero;
            }

            GC.SuppressFinalize(this);
        }

        public static implicit operator IntPtr(Status status) => status._handle;

        public bool Ok => TF_GetCode(_handle) == 0;
        public bool Error => TF_GetCode(_handle) != 0;
    }

    class ApiDefMap : IDisposable
    {
        private IntPtr _handle;

        [DllImport("tensorflow")]
        private static extern unsafe IntPtr TF_NewApiDefMap(IntPtr buffer, IntPtr status);

        [DllImport("tensorflow")]
        private static extern unsafe void TF_DeleteApiDefMap(IntPtr apiDefMap);

        [DllImport("tensorflow")]
        private static extern unsafe void TF_ApiDefMapPut(IntPtr apiDefMap, string text, UInt64 textLen, IntPtr status);

        [DllImport("tensorflow")]
        private static extern unsafe Buffer* TF_ApiDefMapGet(IntPtr apiDefMap, string name, UInt64 nameLen, IntPtr status);

        public unsafe ApiDefMap(Buffer* buffer)
        {
            using (var status = new Status())
            {
                _handle = TF_NewApiDefMap((IntPtr)buffer, status);
                if (status.Error)
                    throw new ArgumentException("Failure to call TF_NewApiDefMap");
            }
        }

        ~ApiDefMap() => Dispose();

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                TF_DeleteApiDefMap(_handle);
                _handle = IntPtr.Zero;
            }

            GC.SuppressFinalize(this);
        }

        public unsafe ApiDef Get(string name)
        {
            using (var status = new Status())
            {
                var buffer = TF_ApiDefMapGet(_handle, name, (UInt64)name.Length, status);

                if (status.Error)
                    return null;

                var bytes = new byte[buffer->length];
                Marshal.Copy(buffer->data, bytes, 0, (int)buffer->length);
                var ms = new MemoryStream(bytes);
                return Serializer.Deserialize<ApiDef>(ms);
            }
        }

        public unsafe bool Put(string text)
        {
            using (var status = new Status())
            {
                TF_ApiDefMapPut(_handle, text, (UInt64)text.Length, status);
                return status.Ok;
            }
        }
    }

    class Generator
    {
        private ApiDefMap _apimap;
        private List<OpDef.AttrDef> _requiredAttrs, _optionalAttrs;
        private bool _haveReturnValue;
        private StreamWriter _output;
        private int _indent = 0;

        [DllImport("tensorflow")]
        private static extern unsafe IntPtr TF_Version();

        [DllImport("tensorflow")]
        private unsafe extern static Buffer* TF_GetAllOpList();

        private string CSharpType(string tfType)
        {
            bool isList = false;
            string cstype;

            if (tfType.StartsWith("list("))
            {
                isList = true;
                tfType = tfType.Substring(5, tfType.Length - 6);
            }

            switch (tfType)
            {
                case "int":
                    cstype = "long";
                    break;
                case "float":
                    cstype = "float";
                    break;
                case "bool":
                    cstype = "bool";
                    break;
                case "type":
                    cstype = "DataType";
                    break;
                case "shape":
                    cstype = "Shape";
                    break;
                case "tensor":
                    cstype = "Tensor";
                    break;
                case "string":
                    cstype = "string";
                    break;
                default:
                    #if DEBUG
                    Console.WriteLine("Unknown TensorFlow type: {0}", tfType);
                    #endif
                    return null;
            }

            return cstype + (isList ? "[]" : "");
        }

        private bool IsReferenceType(string tfType)
        {
            return tfType.StartsWith("list(")
                   || tfType == "tensor"
                   || tfType == "string"
                   || tfType == "shape";
        }

        private string ParamMap(string paramName)
        {
            switch (paramName)
            {
                case "out":
                    return "output";
                case "params":
                    return "parameters";
                case "ref":
                    return "reference";
                case "event":
                    return "event_";
                default:
                    return paramName;
            }
        }

        private bool IsListArg(OpDef.ArgDef argDef)
        {
            return argDef.TypeListAttr != "" || argDef.NumberAttr != "";
        }

        private void SetupArguments(OpDef opDef)
        {
            // Attributes related to the InputArg's type are not exposed.
            var inferredInputArgs = new List<string>();
            _requiredAttrs = new List<OpDef.AttrDef>();
            _optionalAttrs = new List<OpDef.AttrDef>();

            foreach (var argdef in opDef.InputArgs)
            {
                if (argdef.TypeAttr != "")
                    inferredInputArgs.Add(argdef.TypeAttr);

                if (argdef.TypeListAttr != "")
                    inferredInputArgs.Add(argdef.TypeListAttr);

                if (argdef.NumberAttr != "")
                    inferredInputArgs.Add(argdef.NumberAttr);
            }

            foreach (var attr in opDef.Attrs)
            {
                if (!inferredInputArgs.Contains(attr.Name))
                {
                    if (attr.DefaultValue == null)
                        _requiredAttrs.Add(attr);
                    else
                        _optionalAttrs.Add(attr);
                }
            }

            _haveReturnValue = opDef.OutputArgs.Count > 0;
        }

        private string FillArguments(OpDef opDef)
        {
            var sb = new StringBuilder();
            string comma = "";

            foreach (var inArg in opDef.InputArgs)
            {
                string type = "Output!" + (IsListArg(inArg) ? "[]" : "");
                sb.AppendFormat($"{comma}{type} {ParamMap(inArg.Name)}");
                comma = ", ";
            }

            foreach (var attr in _requiredAttrs)
            {
                bool isRefType = IsReferenceType(attr.Type);
                var cstype = CSharpType(attr.Type);
                var cstypeSuffix = isRefType ? "!" : "";
                sb.AppendFormat($"{comma}{cstype}{cstypeSuffix} {ParamMap(attr.Name)}");
                comma = ", ";
            }

            foreach (var attr in _optionalAttrs)
            {
                bool isRefType = IsReferenceType(attr.Type);
                var cstype = CSharpType(attr.Type);
                var cstypeSuffix = isRefType ? "" : "?";

                sb.AppendFormat($"{comma}{cstype}{cstypeSuffix} {ParamMap(attr.Name)} = null");
                comma = ", ";
            }

            if (sb.Length != 0)
                sb.Append(", ");

            return sb.ToString();
        }

        private void Comment(string text)
        {
            if (text == null || text == "")
                return;

            var lines = text.Split('\n');
            var open = true;

            string Quote(string input)
            {
                if (input.IndexOf('`') == -1)
                    return input;

                var sb = new StringBuilder();

                foreach (var c in input)
                {
                    if (c == '`')
                    {
                        sb.Append(open ? "<c>" : "</c>");
                        open = !open;
                    }
                    else
                    {
                        sb.Append(c);
                    }
                }
                return sb.ToString();
            }

            bool blockOpen = true;
            foreach (var line in lines)
            {
                var line2 = line.Trim().Replace("<", "&lt;").Replace(">", "&gt;").Replace("&", "&amp;");

                if (line2.StartsWith("```"))
                {
                    P("///    " + (blockOpen ? "<code>" : "</code>"));
                    blockOpen = !blockOpen;
                    if (line2 == "```python" || line2 == "```c++" || line2 == "```")
                    {
                        continue;
                    }

                    line2 = line2.Substring(3);
                    if (line2.EndsWith("```"))
                    {
                        var line3 = line2.Substring(0, line2.Length - 3);
                        P($"///    {Quote(line3)}");
                        P("///    " + (blockOpen ? "<code>" : "</code>"));
                        blockOpen = !blockOpen;
                        continue;
                    }
                }

                P($"///   {Quote(line2)}");
            }
        }

        private void GenDocs(OpDef opDef)
        {
            var api = _apimap.Get(opDef.Name);
            P("/// <summary>");
            Comment(api.Summary);
            P("/// </summary>");
            foreach (var input in api.InArgs)
            {
                P($"/// <param name=\"{ParamMap(input.Name)}\">");
                Comment(input.Description);
                P($"/// </param>");
            }

            P("/// <param name=\"opName\">");
            P($"///   If specified, the created operation in the graph will be this one, otherwise it will be named '{opDef.Name}'.");
            P("/// </param>");

            foreach (var attr in _optionalAttrs)
            {
                P($"/// <param name=\"{ParamMap(attr.Name)}\">");
                Comment("Optional argument");

                Comment(api.Attrs.Where(x => x.Name == attr.Name).FirstOrDefault().Description);
                P($"/// </param>");
            }

            foreach (var attr in _requiredAttrs)
            {
                P($"/// <param name=\"{ParamMap(attr.Name)}\">");
                Comment(api.Attrs.Where(x => x.Name == attr.Name).FirstOrDefault().Description);
                P($"/// </param>");
            }

            P($"/// <returns>");

            if (_haveReturnValue)
            {
                if (opDef.OutputArgs.Count == 1)
                {
                    Comment(api.OutArgs.First().Description);
                    Comment("The Operation can be fetched from the resulting Output, by fetching the Operation property from the result.");
                }
                else
                {
                    Comment("Returns a tuple with multiple values, as follows:");
                    foreach (var arg in opDef.OutputArgs)
                    {
                        var oapi = api.OutArgs.Where(x => x.Name == arg.Name).FirstOrDefault();
                        Comment(ParamMap(arg.Name) + ": " + oapi.Description);
                    }

                    Comment("The Operation can be fetched from any of the Outputs returned in the tuple values, by fetching the Operation property.");
                }
            }
            else
            {
                Comment("Returns the description of the operation");
            }

            P($"/// </returns>");

            if (!String.IsNullOrEmpty(api.Description))
            {
                P("/// <remarks>");
                Comment(api.Description);
                P("/// </remarks>");
            }
        }

        private void SetAttribute(string type, string attrName, string csAttrName)
        {
            var cstype = CSharpType(type);
            switch (cstype)
            {
                case "long":
                case "long[]":
                case "string":
                case "string[]":
                case "float":
                case "float[]":
                case "bool":
                case "bool[]":
                case "DataType":
                case "DataType[]":
                case "Shape":
                case "Shape[]":
                case "Tensor":
                case "Tensor[]":
                    P($"desc.SetAttr(\"{attrName}\", {csAttrName});");
                    break;
                default:
                    throw new Exception("Unexpected type: " + cstype);
            }
        }

        private void Generate(OpDef opDef)
        {
            SetupArguments(opDef);
            GenDocs(opDef);

            var name = opDef.Name;
            string retType;

            if (_haveReturnValue)
            {
                if (opDef.OutputArgs.Count > 1)
                {
                    var sb = new StringBuilder("(");

                    foreach (var arg in opDef.OutputArgs)
                        sb.AppendFormat("Output{0} {1}, ", IsListArg(arg) ? "[]" : "", ParamMap(arg.Name));

                    sb.Remove(sb.Length - 2, 2);
                    sb.Append(")");
                    retType = sb.ToString();
                }
                else
                {
                    retType = "Output" + (IsListArg(opDef.OutputArgs.First()) ? "[]" : "");
                }
            }
            else
            {
                retType = "Operation";
            }

            P($"public {retType} {name}({FillArguments(opDef)}string opName = null)");
            PI("{");
            bool needStatus = _requiredAttrs.Concat(_optionalAttrs).Any(attr => attr.Type.Contains("Tensor"));
            P($"var desc = new OperationDescription withGraph(this) OpType(\"{opDef.Name}\") OpName(MakeName(\"{opDef.Name}\", opName));");

            foreach (var arg in opDef.InputArgs)
            {
                if (IsListArg(arg))
                    P($"desc.AddInputs({ParamMap(arg.Name)});");
                else
                    P($"desc.AddInput({ParamMap(arg.Name)});");
            }

            P("");
            PI("foreach (Operation control in CurrentDependencies)");
            P("desc.AddControlInput(control);");
            PD("");

            // If we have attributes
            if (_requiredAttrs.Count > 0 || _optionalAttrs.Count > 0)
            {
                foreach (var attr in _requiredAttrs)
                    SetAttribute(attr.Type, attr.Name, ParamMap(attr.Name));

                if (_requiredAttrs.Count > 0)
                    P("");

                foreach (var attr in _optionalAttrs)
                {
                    var reftype = IsReferenceType(attr.Type);
                    var csattr = ParamMap(attr.Name);

                    if (reftype)
                        PI($"if ({csattr} != null)");
                    else
                        PI($"if ({csattr}.HasValue)");

                    // SetAttribute(attr.Type, attr.Name, csattr + (reftype ? "" : ".Value"));
                    SetAttribute(attr.Type, attr.Name, csattr);
                    PD("");
                }
            }

            P("using (var status = new Status())");
            PI("{");
            P("var (success, op) = desc.FinishOperation(status);");
            PI("if(!success)");
            P($"throw new OpCreateException withOpType(\"{opDef.Name}\") Error(status.Message);");
            PD("");

            if (opDef.OutputArgs.Count() > 0)
                P("int _idx = 0;");

            if (opDef.OutputArgs.Any(x => IsListArg(x)))
                P("int _n = 0;");

            foreach (var arg in opDef.OutputArgs)
            {
                if (IsListArg(arg))
                {
                    var outputs = new StringBuilder();
                    P($"_n = op.GetOutputListLength(\"{ParamMap(arg.Name)}\");");
                    P($"var {ParamMap(arg.Name)} = new Output[_n];");
                    PI("for (int i = 0; i < _n; i++)");
                    P($"{ParamMap(arg.Name)} [i] = new Output withOp(op) Index(_idx++);");
                    PD("\n");
                }
                else
                {
                    P($"var {ParamMap(arg.Name)} = new Output withOp(op) Index(_idx++);");
                }
            }

            if (_haveReturnValue)
            {
                if (opDef.OutputArgs.Count == 1)
                    P($"return {ParamMap(opDef.OutputArgs.First().Name)};");
                else
                    P("return (" + opDef.OutputArgs.Select(x => ParamMap(x.Name)).Aggregate((i, j) => (i + ", " + j)) + ");");
            }
            else
            {
                P("return op;");
            }

            PD("}");
            PD("}\n");
        }

        private MemoryStream GetOpsList()
        {
            unsafe
            {
                Buffer* buffer = TF_GetAllOpList();
                _apimap = new ApiDefMap(buffer);
                var ret = new byte[(int)buffer->length];
                Marshal.Copy(buffer->data, ret, 0, (int)buffer->length);
                return new MemoryStream(ret);
            }
        }

        private void UpdateApiDefs(string[] apiDefDirs)
        {
            foreach (var dir in apiDefDirs)
            {
                foreach (var f in Directory.GetFiles(dir))
                {
                    var s = File.ReadAllText(f);
                    _apimap.Put(s);
                }
            }
        }

        private void WriteLicense()
        {
            P(@"/*");
            var licText = File.ReadAllText("./LICENSE");
            _output.Write(licText);
            P(@"*/");
        }

        private void Run(string[] apiDefDirs)
        {
            _output = File.CreateText("../../Classes/OpsGenerated.cs");
            var opDefList = Serializer.Deserialize<List<OpDef>>(GetOpsList());
            UpdateApiDefs(apiDefDirs);
            WriteLicense();
            P("");
            P("using RemObjects.Elements.System;");
            P("using TensorFlow.Island.Api;\n");

            P("namespace TensorFlow.Island.Classes");
            PI("{");
            P("public partial class Graph");
            PI("{\n");

            foreach (var opDef in (from o in opDefList orderby o.Name select o))
            {
                // Skip internal operations
                if (opDef.Name.StartsWith("_"))
                    continue;

                // Ignore functions where we lack a C# type mapping
                if (opDef.Attrs.Any(a => CSharpType(a.Type) == null))
                {
                    var attr = opDef.Attrs.First(a => CSharpType(a.Type) == null);
                    Console.WriteLine($"Skip Op: {opDef.Name} due to attribute ({attr.Type} {attr.Name}) lacking a mapping to C#");
                    continue;
                }

                // Undocumented operation, perhaps we should not surface
                if (_apimap.Get(opDef.Name).Summary.Equals(""))
                    continue;

                Generate(opDef);
            }

            PD("}");
            PD("}");
            _output.Close();
        }

        private void PI(string fmt, params object[] args)
        {
            P(fmt, args);
            _indent++;
        }

        private void PD(string fmt, params object[] args)
        {
            _indent--;
            P(fmt, args);
        }

        private void P(string fmt, params object[] args)
        {
            for (int i = 0; i < _indent; i++)
                _output.Write("    ");

            if (args.Length == 0)
                _output.WriteLine(fmt);
            else
                _output.WriteLine(fmt, args);
        }

        public static void Main(string[] args)
        {
            var ver = Version;
            Console.WriteLine($"Getting Api definition for TensorFlow {ver}");

            if (Marshal.SizeOf(typeof(IntPtr)) != 8)
                throw new Exception("This program only supports 64bit mode.");

            if (args.Length == 0)
            {
                var apiDefDir = $"../../../../TensorFlow/tensorflow/core/api_def/base_api";
                args = new string[] { apiDefDir };
            }

            new Generator().Run(apiDefDirs: args);
            Console.WriteLine("Please press any key to exit.");
            Console.ReadLine();
        }

        public static string Version => Marshal.PtrToStringAnsi(TF_Version());
    }
}
