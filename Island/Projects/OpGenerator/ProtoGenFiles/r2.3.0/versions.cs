// This file was generated by a tool; you should avoid making direct changes.
// Consider using 'partial classes' to extend these types
// Input: versions.proto

#pragma warning disable CS1591, CS0612, CS3021, IDE1006
namespace Tensorflow
{

    [global::ProtoBuf.ProtoContract()]
    public partial class VersionDef : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"producer")]
        public int Producer { get; set; }

        [global::ProtoBuf.ProtoMember(2, Name = @"min_consumer")]
        public int MinConsumer { get; set; }

        [global::ProtoBuf.ProtoMember(3, Name = @"bad_consumers", IsPacked = true)]
        public int[] BadConsumers { get; set; }

    }

}

#pragma warning restore CS1591, CS0612, CS3021, IDE1006
