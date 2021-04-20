// <auto-generated>
//   This file was generated by a tool; you should avoid making direct changes.
//   Consider using 'partial classes' to extend these types
//   Input: function.proto
// </auto-generated>

#region Designer generated code
#pragma warning disable CS0612, CS0618, CS1591, CS3021, IDE0079, IDE1006, RCS1036, RCS1057, RCS1085, RCS1192
namespace Tensorflow
{

    [global::ProtoBuf.ProtoContract()]
    public partial class FunctionDefLibrary : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"function")]
        public global::System.Collections.Generic.List<FunctionDef> Functions { get; } = new global::System.Collections.Generic.List<FunctionDef>();

        [global::ProtoBuf.ProtoMember(2, Name = @"gradient")]
        public global::System.Collections.Generic.List<GradientDef> Gradients { get; } = new global::System.Collections.Generic.List<GradientDef>();

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class FunctionDef : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"signature")]
        public OpDef Signature { get; set; }

        [global::ProtoBuf.ProtoMember(5, Name = @"attr")]
        [global::ProtoBuf.ProtoMap]
        public global::System.Collections.Generic.Dictionary<string, AttrValue> Attrs { get; } = new global::System.Collections.Generic.Dictionary<string, AttrValue>();

        [global::ProtoBuf.ProtoMember(7)]
        [global::ProtoBuf.ProtoMap]
        public global::System.Collections.Generic.Dictionary<uint, FunctionDef.ArgAttrs> arg_attr { get; } = new global::System.Collections.Generic.Dictionary<uint, FunctionDef.ArgAttrs>();

        [global::ProtoBuf.ProtoMember(8, Name = @"resource_arg_unique_id")]
        [global::ProtoBuf.ProtoMap]
        public global::System.Collections.Generic.Dictionary<uint, uint> ResourceArgUniqueIds { get; } = new global::System.Collections.Generic.Dictionary<uint, uint>();

        [global::ProtoBuf.ProtoMember(3, Name = @"node_def")]
        public global::System.Collections.Generic.List<NodeDef> NodeDefs { get; } = new global::System.Collections.Generic.List<NodeDef>();

        [global::ProtoBuf.ProtoMember(4, Name = @"ret")]
        [global::ProtoBuf.ProtoMap]
        public global::System.Collections.Generic.Dictionary<string, string> Rets { get; } = new global::System.Collections.Generic.Dictionary<string, string>();

        [global::ProtoBuf.ProtoMember(6, Name = @"control_ret")]
        [global::ProtoBuf.ProtoMap]
        public global::System.Collections.Generic.Dictionary<string, string> ControlRets { get; } = new global::System.Collections.Generic.Dictionary<string, string>();

        [global::ProtoBuf.ProtoContract()]
        public partial class ArgAttrs : global::ProtoBuf.IExtensible
        {
            private global::ProtoBuf.IExtension __pbn__extensionData;
            global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
                => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

            [global::ProtoBuf.ProtoMember(1, Name = @"attr")]
            [global::ProtoBuf.ProtoMap]
            public global::System.Collections.Generic.Dictionary<string, AttrValue> Attrs { get; } = new global::System.Collections.Generic.Dictionary<string, AttrValue>();

        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class GradientDef : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"function_name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string FunctionName { get; set; } = "";

        [global::ProtoBuf.ProtoMember(2, Name = @"gradient_func")]
        [global::System.ComponentModel.DefaultValue("")]
        public string GradientFunc { get; set; } = "";

    }

}

#pragma warning restore CS0612, CS0618, CS1591, CS3021, IDE0079, IDE1006, RCS1036, RCS1057, RCS1085, RCS1192
#endregion
