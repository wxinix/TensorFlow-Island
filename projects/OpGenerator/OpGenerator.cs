//
// This is the driver for the operation generator, this takes data that
// is provided by the Tensorflow runtime to produce strongly-typed and
// high level methods on the TFGraph class.
//
// The result is generated into a partial class that is lined with the
// main TensorFlowSharp library
//
// Authors:
//   Miguel de Icaza
//
// Copyright 2017, the year of downfall, Microsoft Inc
//
#pragma warning disable RECS0063 // Warns when a culture-aware 'StartsWith' call is used by default.

using System;
using System.Collections.Generic;
using System.IO;
using ProtoBuf;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using Tensorflow;

class ApiDefMap : IDisposable 
{
	public class Status : IDisposable
	{
		[DllImport ("tensorflow")]
		static extern unsafe IntPtr TF_NewStatus ();

		[DllImport ("tensorflow")]
		internal static extern unsafe void TF_DeleteStatus (IntPtr status);

		[DllImport ("tensorflow")]
		static extern unsafe int TF_GetCode (IntPtr s);

		IntPtr handle;
		public Status ()
		{
			handle = TF_NewStatus ();
		}

		void IDisposable.Dispose ()
		{
			TF_DeleteStatus (handle);
			handle = IntPtr.Zero;
		}

		public bool Ok => TF_GetCode (handle) == 0;
		public bool Error => TF_GetCode (handle) != 0;

		public static implicit operator IntPtr (Status s)
		{
			return s.handle;
		}
	}

	[DllImport ("tensorflow")]
	unsafe extern static IntPtr TF_NewApiDefMap (IntPtr buffer, IntPtr status);

	[DllImport ("tensorflow")]
	static extern void TF_DeleteApiDefMap (IntPtr handle);

	[DllImport ("tensorflow")]
	static extern void TF_ApiDefMapPut (IntPtr handle, string text, IntPtr textLen, IntPtr status);

	[DllImport ("tensorflow")]
	unsafe static extern OpGenerator.LLBuffer *TF_ApiDefMapGet (IntPtr handle, string name, IntPtr nameLen, IntPtr status);

	IntPtr handle;

	unsafe public ApiDefMap (OpGenerator.LLBuffer* buffer)
	{
		using (var status = new Status ()) {
			handle = TF_NewApiDefMap ((IntPtr)buffer, status);

			if (status.Error)
				throw new ArgumentException ("Failure to call TF_NewApiDefMap");
		}
	}

	void IDisposable.Dispose ()
	{
		Dispose (true);
		GC.SuppressFinalize (this);
	}

	~ApiDefMap ()
	{
		Dispose (false);
	}

	void Dispose (bool disposing)
	{
		if (disposing) {
			if (handle != IntPtr.Zero) {
				TF_DeleteApiDefMap (handle);
				handle = IntPtr.Zero;
			}
		}
	}

	public unsafe ApiDef Get (string name)
	{
		using (var status = new Status ()) {
			var ptr = TF_ApiDefMapGet (handle, name, (IntPtr)name.Length, status);
			if (status.Error)
				return null;
			var ret = new byte [(int)ptr->length];
			Marshal.Copy (ptr->data, ret, 0, (int)ptr->length);
			var str = new MemoryStream (ret);
			return Serializer.Deserialize<ApiDef> (str);
		}
	}

	public unsafe bool Put (string text)
	{
		using (var status = new Status ()) {
			TF_ApiDefMapPut (handle, text, (IntPtr)text.Length, status);
			if (status.Error)
				return false;
			return true;
		}
	}
}

class OpGenerator
{
	[DllImport ("tensorflow")]
	static extern unsafe IntPtr TF_Version ();

	public static string GetVersion ()
	{
		return Marshal.PtrToStringAnsi (TF_Version ());
	}

	//
	// Maps a TensorFlow type to a C# type
	//
	string CSharpType (string tfType)
	{
		bool list = false;
		string cstype;

		if (tfType.StartsWith ("list(")) {
			list = true;
			tfType = tfType.Substring (5, tfType.Length - 6);
		}
		switch (tfType) {
		case "int":
			cstype = "long"; break;
		case "float":
			cstype = "float"; break;
		case "bool":
			cstype = "bool"; break;
		case "type":
			cstype = "TFDataType"; break;
		case "shape":
			cstype = "TFShape"; break;
		case "tensor":
			cstype = "TFTensor"; break;
		case "string":
			cstype = "string"; break;
		default:
			Console.WriteLine ("Unknown data TensorFlow type: {0}", tfType);
			return null;
		}

		return cstype + (list ? "[]" : "");
	}

	bool IsReferenceType (string tfType)
	{
		if (tfType.StartsWith ("list("))
			return true;
		if (tfType == "tensor" || tfType == "string" || tfType == "shape")
			return true;
		return false;
	}

	// Maps a parameter name to a C# acceptable name, to avoid clashes with 
	// language keywords
	string ParamMap (string paramName)
	{
		switch (paramName) {
		case "out":
			return "output";
		case "params":
			return "parameters";
		case "ref":
			return "reference";
		case "event":
			return "evnt";
		}
		return paramName;
	}

	// Determines if the specified ArgDef represents a TensorFlow list
	bool IsListArg (OpDef.ArgDef arg)
	{
		return arg.TypeListAttr != "" || arg.NumberAttr != "";
	}

	// 
	// These values are the result of calling SetupArguments
	//
	List<OpDef.AttrDef> required_attrs, optional_attrs;
	bool have_return_value;

	void SetupArguments (OpDef def)
	{
		// Attributes related to the InputArg's type are inferred automatically
		// and are not exposed to the client.
		var inferred_input_args = new Dictionary<string, bool> ();
		required_attrs = new List<OpDef.AttrDef> ();
		optional_attrs = new List<OpDef.AttrDef> ();

		foreach (var argdef in def.InputArgs) {
			if (argdef.TypeAttr != "")
				inferred_input_args [argdef.TypeAttr] = true;
			else if (argdef.TypeListAttr != "")
				inferred_input_args [argdef.TypeListAttr] = true;
			if (argdef.NumberAttr != "")
				inferred_input_args [argdef.NumberAttr] = true;
		}
		foreach (var attr in def.Attrs) {
			if (inferred_input_args.ContainsKey (attr.Name))
				continue;
			if (attr.DefaultValue == null)
				required_attrs.Add (attr);
			else
				optional_attrs.Add (attr);
		}
		have_return_value = def.OutputArgs.Count > 0;
	}

	// Generates arguments:
	//   * Input arguments (TFOutput or TFOutput [])
	//   * All required attributes
	//   * variadic optional arguments
	string FillArguments (OpDef def)
	{
		var sb = new StringBuilder ();
		string comma = "";
		foreach (var inarg in def.InputArgs) {
			string type = "TFOutput" + (IsListArg (inarg) ? "[]" : "");

			sb.AppendFormat ($"{comma}{type} {ParamMap (inarg.Name)}");
			comma = ", ";
		}
		foreach (var attr in required_attrs) {
			sb.AppendFormat ($"{comma}{CSharpType (attr.Type)} {ParamMap (attr.Name)}");
			comma = ", ";
		}

#if false
		if (!return_is_tfoutput) {
			foreach (var arg in def.OutputArg) {
				string type = "TFOutput" + (IsListArg (arg) ? "[]" : "");

				sb.AppendFormat ($"{comma}ref {type} {ParamMap (arg.Name)}");
				comma = ", ";
			}
		}
#endif
		foreach (var attr in optional_attrs) {
			bool reftype = IsReferenceType (attr.Type);
			var cstype = CSharpType (attr.Type);
			var cstypesuffix = reftype ? "" : "?";

			sb.AppendFormat ($"{comma}{cstype}{cstypesuffix} {attr.Name} = null");
			comma = ", ";
		}
		if (sb.Length != 0)
			sb.Append (", ");
		return sb.ToString ();
	}

	void Comment (string text)
	{
		if (text == null || text == "")
			return;
		var lines = text.Split ('\n');
		var open = true;

		string Quote (string input)
		{
			var p = input.IndexOf ('`');
			if (p == -1)
				return input;
			var res = new StringBuilder ();
			foreach (var c in input) {
				if (c == '`') {
					res.Append (open ? "<c>" : "</c>");
					open = !open;
				} else
					res.Append (c);
			}
			return res.ToString ();
		}

		bool blockOpen = true;
		foreach (var line in lines) {
			if (line.IndexOf ("in image height coordinates.") != -1) {
				Console.WriteLine ("Hello");
			}

			var line2 = line.Trim ().Replace ("<", "&lt;").Replace (">", "&gt;").Replace ("&", "&amp;");

			if (line2.StartsWith ("```")){
				p ("///    " + (blockOpen ? "<code>" : "</code>"));
				blockOpen = !blockOpen;
				if (line2 == "```python" || line2 == "```c++" || line2 == "```")
					continue;
				// Handle some broken comments in the api specs, they sometimes missuse the 

				line2 = line2.Substring (3);
				if (line2.EndsWith ("```")){
					var line3 = line2.Substring (0, line2.Length - 3);
					p ($"///    {Quote (line3)}");
					p ("///    " + (blockOpen ? "<code>" : "</code>"));
					blockOpen = !blockOpen;
					continue;
				}
			} 
			p ($"///   {Quote (line2)}");

		}
	}


	// Produces the C# inline documentation
	void GenDocs (OpDef oper)
	{
		var api = apimap.Get (oper.Name);
		p ("/// <summary>");
		Comment (api.Summary);
		p ("/// </summary>");
		foreach (var input in api.InArgs) {
			p ($"/// <param name=\"{ParamMap (input.Name)}\">");
			Comment (input.Description);
			p ($"/// </param>");
		}
#if DOCS
		if (!return_is_tfoutput) {
			foreach (var attr in oper.OutputArg) {
				if (String.IsNullOrEmpty (attr.description))
					continue;
				p ($"/// <param name=\"{ParamMap (attr.Name)}\">");
				Comment (attr.description);
				p ($"/// </param>");
			}
		}
#endif
		p ("/// <param name=\"operName\">");
		p ($"///   If specified, the created operation in the graph will be this one, otherwise it will be named '{oper.Name}'.");
		p ("/// </param>");
		foreach (var attr in optional_attrs) {
			p ($"/// <param name=\"{ParamMap (attr.Name)}\">");
			Comment ("Optional argument");

			Comment (api.Attrs.Where (x=>x.Name == attr.Name).FirstOrDefault ().Description);
			p ($"/// </param>");
		}
		foreach (var attr in required_attrs) {
			p ($"/// <param name=\"{ParamMap (attr.Name)}\">");
			Comment (api.Attrs.Where (x=>x.Name == attr.Name).FirstOrDefault ().Description);
			p ($"/// </param>");
		}
		p ($"/// <returns>");
		if (have_return_value) {
			if (oper.OutputArgs.Count == 1) {
				Comment (api.OutArgs.First ().Description);
				Comment ("The TFOperation can be fetched from the resulting TFOutput, by fetching the Operation property from the result.");
			} else {
				Comment ("Returns a tuple with multiple values, as follows:");
				foreach (var arg in oper.OutputArgs) {
					var oapi = api.OutArgs.Where (x => x.Name == arg.Name).FirstOrDefault ();
					Comment (ParamMap (arg.Name) + ": " + oapi.Description);
				}

				Comment ("The TFOperation can be fetched from any of the TFOutputs returned in the tuple values, by fetching the Operation property.");
			}
		} else {
			Comment ("Returns the description of the operation");
		}
		p ($"/// </returns>");

		if (!String.IsNullOrEmpty (api.Description)) {
			p ("/// <remarks>");
			Comment (api.Description);
			p ("/// </remarks>");
		}
	}

	void SetAttribute (string type, string attrName, string csAttrName)
	{
		if (type == "shape") {
			p ($"desc.SetAttrShape (\"{attrName}\", {csAttrName});");
			return;
		}
		if (type.StartsWith ("list(shape")) {
			p ($"desc.SetAttrShape (\"{attrName}\", {csAttrName});");
			return;
		}

		var cstype = CSharpType (type);
		switch (cstype) {
		case "long":
		case "long[]":
		case "string":
		case "string[]":
		case "float":
		case "float[]":
		case "bool":
		case "bool[]":
			p ($"desc.SetAttr (\"{attrName}\", {csAttrName});");
			break;
		case "TFDataType":
		case "TFDataType[]":
			p ($"desc.SetAttrType (\"{attrName}\", {csAttrName});");
			break;

		// This should pass the cstatus, but requires the 
		// function to take a TFStatus as well, so need to weave that
		// in the parameters
		case "TFTensor":
		case "TFTensor[]":
			p ($"desc.SetAttr (\"{attrName}\", {csAttrName} /* cstatus */);");
			break;
		default:
			throw new Exception ("Unexpected type: " + cstype);
		}
	}

	/// <summary>
	/// Generate the specified oper.
	/// </summary>
	/// <param name="oper">Oper.</param>
	void Generate (OpDef oper)
	{

		SetupArguments (oper);
		GenDocs (oper);

		var name = oper.Name;
		string retType;

		if (have_return_value) {
			if (oper.OutputArgs.Count > 1) {
				var rb = new StringBuilder ("(");
				foreach (var arg in oper.OutputArgs) {
					rb.AppendFormat ("TFOutput{0} {1}, ", IsListArg (arg) ? "[]" : "", ParamMap (arg.Name));
				}
				rb.Remove (rb.Length - 2, 2);
				rb.Append (")");
				retType = rb.ToString ();
			} else 
				retType = "TFOutput" + (IsListArg (oper.OutputArgs.First ()) ? "[]" : "");
		} else
			retType = "TFOperation";
		
		p ($"public {retType} {name} ({FillArguments(oper)}string operName = null)");
		pi ("{");
		bool needStatus = required_attrs.Concat (optional_attrs).Any (attr => attr.Type.Contains ("TFTensor"));
		p ($"var desc = new TFOperationDesc (this, \"{oper.Name}\", MakeName (\"{oper.Name}\", operName));");
		foreach (var arg in oper.InputArgs) {
			if (IsListArg (arg))
				p ($"desc.AddInputs ({ParamMap (arg.Name)});");
			   else
				p ($"desc.AddInput ({ParamMap (arg.Name)});");
		}

		pi ("foreach ( TFOperation control in CurrentDependencies )");
		p ("desc.AddControlInput (control);");
		pd ("");

		// If we have attributes
		if (required_attrs.Count > 0 || optional_attrs.Count > 0) {
			foreach (var attr in required_attrs) {
				SetAttribute (attr.Type, attr.Name, ParamMap (attr.Name));
			}

			foreach (var attr in optional_attrs) {
				var reftype = IsReferenceType (attr.Type);
				var csattr = ParamMap (attr.Name);
				if (reftype)
					pi ($"if ({csattr} != null)");
				else
					pi ($"if ({csattr}.HasValue)");
				SetAttribute (attr.Type, attr.Name, csattr + (reftype ? "" : ".Value"));
				pd ("");

			}
		}

		p ("var op = desc.FinishOperation ();");
		if (oper.OutputArgs.Count () > 0)
			p ("int _idx = 0;");
		if (oper.OutputArgs.Any (x => IsListArg (x)))
			p ("int _n = 0;");
		foreach (var arg in oper.OutputArgs) {
			if (IsListArg (arg)) {
				var outputs = new StringBuilder ();
				p ($"_n = op.OutputListLength (\"{ParamMap (arg.Name)}\");");
				p ($"var {ParamMap (arg.Name)} = new TFOutput [_n];");
				pi ("for (int i = 0; i < _n; i++)");
				p ($"{ParamMap (arg.Name)} [i] = new TFOutput (op, _idx++);");
				pd ("");
			} else {
				p ($"var {ParamMap (arg.Name)} = new TFOutput (op, _idx++);");
			}
		}

		if (have_return_value) {
			if (oper.OutputArgs.Count == 1) {
				p ($"return {ParamMap (oper.OutputArgs.First ().Name)};");
			} else {
				;
				p ("return (" + oper.OutputArgs.Select (x => ParamMap (x.Name)).Aggregate ((i, j) => (i + ", " + j)) + ");");
			}
		} else {
			p ("return op;");
		}
		pd ("}\n");
	}

	[StructLayout (LayoutKind.Sequential)]
	internal struct LLBuffer
	{
		internal IntPtr data;
		internal IntPtr length;
		internal IntPtr data_deallocator;
	}

	[DllImport ("tensorflow")]
	unsafe extern static LLBuffer *TF_GetAllOpList ();
	ApiDefMap apimap;

	MemoryStream GetOpsList ()
	{
		unsafe
		{
			LLBuffer* ptr = TF_GetAllOpList ();
			apimap = new ApiDefMap (ptr);
			var ret = new byte [(int)ptr->length];
			Marshal.Copy (ptr->data, ret, 0, (int)ptr->length);
			return new MemoryStream (ret);
		}
	}

	// Incorporates out-of-band data into the API definitions that we pulled out of GetAllOpList
	void UpdateApis (string [] dirs)
	{
		foreach (var dir in dirs) {
			foreach (var f in Directory.GetFiles (dir)) {
				var s = File.ReadAllText (f);
				apimap.Put (s);
			}
		}
	}

	void Run (string [] dirs)
	{
		output = File.CreateText ("../../../Src/Classes/GeneratedOps.cs");
	     	var operations = Serializer.Deserialize<List<OpDef>> (GetOpsList ());
		UpdateApis (dirs);
		p ("using System;\n");

		pi ("namespace TensorFlow {");
		pi ("public partial class TFGraph {");
		foreach (var oper in (from o in operations orderby o.Name select o)){
			// Skip internal operations
			if (oper.Name.StartsWith ("_"))
				continue;

			// Ignore functions where we lack a C# type mapping
			if (oper.Attrs.Any (attr => CSharpType (attr.Type) == null)) {
				var attr = oper.Attrs.First (a => CSharpType (a.Type) == null);

				Console.WriteLine ($"SkipTYPE: {oper.Name} due to attribute ({attr.Type} {attr.Name}) lacking a mapping to C#");
				continue;
			}

#if false
			// Ignore reference types as well (per go's binding)
			if (oper.input_arg.Any (ia => ia.is_ref)) {
				var pars = String.Join (", ", oper.input_arg.Where (x => x.is_ref).Select (x => $"{x.Type} {x.Name}"));
				Console.WriteLine ($"SkipInREF: {oper.Name} parameters with is_ref: {pars}");
				continue;
			}

			// Ignore reference types as well (per go's binding)
			if (oper.OutputArg.Any (ia => ia.is_ref)) {
				var pars = String.Join (", ", oper.OutputArg.Where (x => x.is_ref).Select (x => $"{x.Type} {x.Name}"));
				var all = String.Join (", ", oper.input_arg.Select (x => $"{x.Type} {x.Name}"));
				Console.WriteLine ($"SkipOutREF: {oper.Name} parameters with is_ref: {pars} all: {all}");

				continue;
			}
#endif
			var def = apimap.Get (oper.Name);

			// Undocumented operation, perhaps we should not surface
			if (def.Summary == "")
				continue;

			Generate (oper);
		}
		pd ("}");
		pd ("}");
		output.Close ();
	}

	// The output file
	StreamWriter output;

	int indent = 0;

	// Convenience methods to generate output
	void pi (string fmt, params object [] args)
	{
		p (fmt, args);
		indent++;
	}

	void pd (string fmt, params object [] args)
	{
		indent--;
		p (fmt, args);
	}

	void p (string fmt, params object [] args)
	{
		for (int i = 0; i < indent; i++)
		     output.Write ("\t");
		if (args.Length == 0)
			output.WriteLine (fmt);
		else
			output.WriteLine (fmt, args);
	}

	public static void Main (string [] args)
	{
		Console.WriteLine ("Getting code for {0}", GetVersion ());
		if (Marshal.SizeOf (typeof (IntPtr)) != 8)
			throw new Exception ("Need to run in 64");
		if (args.Length == 0)
			args = new string [] { "../../../TensorFlow/tensorflow-r1.15/tensorflow/core/api_def/base_api" };



		new OpGenerator ().Run (args);
	}
}
