// This file was generated by a tool; you should avoid making direct changes.
// Consider using 'partial classes' to extend these types
// Input: reader_base.proto

#pragma warning disable CS1591, CS0612, CS3021, IDE1006
namespace Tensorflow
{

    [global::ProtoBuf.ProtoContract()]
    public partial class ReaderBaseState : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"work_started")]
        public long WorkStarted { get; set; }

        [global::ProtoBuf.ProtoMember(2, Name = @"work_finished")]
        public long WorkFinished { get; set; }

        [global::ProtoBuf.ProtoMember(3, Name = @"num_records_produced")]
        public long NumRecordsProduced { get; set; }

        [global::ProtoBuf.ProtoMember(4, Name = @"current_work")]
        public byte[] CurrentWork { get; set; }

    }

}

#pragma warning restore CS1591, CS0612, CS3021, IDE1006
