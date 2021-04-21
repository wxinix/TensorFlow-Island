// <auto-generated>
//   This file was generated by a tool; you should avoid making direct changes.
//   Consider using 'partial classes' to extend these types
//   Input: attr_value.proto
// </auto-generated>

#region Designer generated code
#pragma warning disable CS0612, CS0618, CS1591, CS3021, IDE0079, IDE1006, RCS1036, RCS1057, RCS1085, RCS1192
namespace Tensorflow
{

    [global::ProtoBuf.ProtoContract()]
    public partial class AttrValue : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(2, Name = @"s")]
        public byte[] S
        {
            get => __pbn__value.Is(2) ? ((byte[])__pbn__value.Object) : default;
            set => __pbn__value = new global::ProtoBuf.DiscriminatedUnion64Object(2, value);
        }
        public bool ShouldSerializeS() => __pbn__value.Is(2);
        public void ResetS() => global::ProtoBuf.DiscriminatedUnion64Object.Reset(ref __pbn__value, 2);

        private global::ProtoBuf.DiscriminatedUnion64Object __pbn__value;

        [global::ProtoBuf.ProtoMember(3, Name = @"i")]
        public long I
        {
            get => __pbn__value.Is(3) ? __pbn__value.Int64 : default;
            set => __pbn__value = new global::ProtoBuf.DiscriminatedUnion64Object(3, value);
        }
        public bool ShouldSerializeI() => __pbn__value.Is(3);
        public void ResetI() => global::ProtoBuf.DiscriminatedUnion64Object.Reset(ref __pbn__value, 3);

        [global::ProtoBuf.ProtoMember(4, Name = @"f")]
        public float F
        {
            get => __pbn__value.Is(4) ? __pbn__value.Single : default;
            set => __pbn__value = new global::ProtoBuf.DiscriminatedUnion64Object(4, value);
        }
        public bool ShouldSerializeF() => __pbn__value.Is(4);
        public void ResetF() => global::ProtoBuf.DiscriminatedUnion64Object.Reset(ref __pbn__value, 4);

        [global::ProtoBuf.ProtoMember(5, Name = @"b")]
        public bool B
        {
            get => __pbn__value.Is(5) ? __pbn__value.Boolean : default;
            set => __pbn__value = new global::ProtoBuf.DiscriminatedUnion64Object(5, value);
        }
        public bool ShouldSerializeB() => __pbn__value.Is(5);
        public void ResetB() => global::ProtoBuf.DiscriminatedUnion64Object.Reset(ref __pbn__value, 5);

        [global::ProtoBuf.ProtoMember(6, Name = @"type")]
        public DataType Type
        {
            get => __pbn__value.Is(6) ? ((DataType)__pbn__value.Int32) : default;
            set => __pbn__value = new global::ProtoBuf.DiscriminatedUnion64Object(6, (int)value);
        }
        public bool ShouldSerializeType() => __pbn__value.Is(6);
        public void ResetType() => global::ProtoBuf.DiscriminatedUnion64Object.Reset(ref __pbn__value, 6);

        [global::ProtoBuf.ProtoMember(7, Name = @"shape")]
        public TensorShapeProto Shape
        {
            get => __pbn__value.Is(7) ? ((TensorShapeProto)__pbn__value.Object) : default;
            set => __pbn__value = new global::ProtoBuf.DiscriminatedUnion64Object(7, value);
        }
        public bool ShouldSerializeShape() => __pbn__value.Is(7);
        public void ResetShape() => global::ProtoBuf.DiscriminatedUnion64Object.Reset(ref __pbn__value, 7);

        [global::ProtoBuf.ProtoMember(8, Name = @"tensor")]
        public TensorProto Tensor
        {
            get => __pbn__value.Is(8) ? ((TensorProto)__pbn__value.Object) : default;
            set => __pbn__value = new global::ProtoBuf.DiscriminatedUnion64Object(8, value);
        }
        public bool ShouldSerializeTensor() => __pbn__value.Is(8);
        public void ResetTensor() => global::ProtoBuf.DiscriminatedUnion64Object.Reset(ref __pbn__value, 8);

        [global::ProtoBuf.ProtoMember(1, Name = @"list")]
        public ListValue List
        {
            get => __pbn__value.Is(1) ? ((ListValue)__pbn__value.Object) : default;
            set => __pbn__value = new global::ProtoBuf.DiscriminatedUnion64Object(1, value);
        }
        public bool ShouldSerializeList() => __pbn__value.Is(1);
        public void ResetList() => global::ProtoBuf.DiscriminatedUnion64Object.Reset(ref __pbn__value, 1);

        [global::ProtoBuf.ProtoMember(10, Name = @"func")]
        public NameAttrList Func
        {
            get => __pbn__value.Is(10) ? ((NameAttrList)__pbn__value.Object) : default;
            set => __pbn__value = new global::ProtoBuf.DiscriminatedUnion64Object(10, value);
        }
        public bool ShouldSerializeFunc() => __pbn__value.Is(10);
        public void ResetFunc() => global::ProtoBuf.DiscriminatedUnion64Object.Reset(ref __pbn__value, 10);

        [global::ProtoBuf.ProtoMember(9, Name = @"placeholder")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Placeholder
        {
            get => __pbn__value.Is(9) ? ((string)__pbn__value.Object) : "";
            set => __pbn__value = new global::ProtoBuf.DiscriminatedUnion64Object(9, value);
        }
        public bool ShouldSerializePlaceholder() => __pbn__value.Is(9);
        public void ResetPlaceholder() => global::ProtoBuf.DiscriminatedUnion64Object.Reset(ref __pbn__value, 9);

        [global::ProtoBuf.ProtoContract()]
        public partial class ListValue : global::ProtoBuf.IExtensible
        {
            private global::ProtoBuf.IExtension __pbn__extensionData;
            global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
                => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

            [global::ProtoBuf.ProtoMember(2, Name = @"s")]
            public global::System.Collections.Generic.List<byte[]> S { get; } = new global::System.Collections.Generic.List<byte[]>();

            [global::ProtoBuf.ProtoMember(3, Name = @"i", IsPacked = true)]
            public long[] I { get; set; }

            [global::ProtoBuf.ProtoMember(4, Name = @"f", IsPacked = true)]
            public float[] F { get; set; }

            [global::ProtoBuf.ProtoMember(5, Name = @"b", IsPacked = true)]
            public bool[] B { get; set; }

            [global::ProtoBuf.ProtoMember(6, Name = @"type", IsPacked = true)]
            public global::System.Collections.Generic.List<DataType> Types { get; } = new global::System.Collections.Generic.List<DataType>();

            [global::ProtoBuf.ProtoMember(7, Name = @"shape")]
            public global::System.Collections.Generic.List<TensorShapeProto> Shapes { get; } = new global::System.Collections.Generic.List<TensorShapeProto>();

            [global::ProtoBuf.ProtoMember(8, Name = @"tensor")]
            public global::System.Collections.Generic.List<TensorProto> Tensors { get; } = new global::System.Collections.Generic.List<TensorProto>();

            [global::ProtoBuf.ProtoMember(9, Name = @"func")]
            public global::System.Collections.Generic.List<NameAttrList> Funcs { get; } = new global::System.Collections.Generic.List<NameAttrList>();

        }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class NameAttrList : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Name { get; set; } = "";

        [global::ProtoBuf.ProtoMember(2, Name = @"attr")]
        [global::ProtoBuf.ProtoMap]
        public global::System.Collections.Generic.Dictionary<string, AttrValue> Attrs { get; } = new global::System.Collections.Generic.Dictionary<string, AttrValue>();

    }

}

#pragma warning restore CS0612, CS0618, CS1591, CS3021, IDE0079, IDE1006, RCS1036, RCS1057, RCS1085, RCS1192
#endregion