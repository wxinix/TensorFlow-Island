// This file was generated by a tool; you should avoid making direct changes.
// Consider using 'partial classes' to extend these types
// Input: log_memory.proto

#pragma warning disable CS1591, CS0612, CS3021, IDE1006
namespace Tensorflow
{

    [global::ProtoBuf.ProtoContract()]
    public partial class MemoryLogStep : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"step_id")]
        public long StepId { get; set; }

        [global::ProtoBuf.ProtoMember(2, Name = @"handle")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Handle { get; set; } = "";

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class MemoryLogTensorAllocation : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"step_id")]
        public long StepId { get; set; }

        [global::ProtoBuf.ProtoMember(2, Name = @"kernel_name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string KernelName { get; set; } = "";

        [global::ProtoBuf.ProtoMember(3, Name = @"tensor")]
        public TensorDescription Tensor { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class MemoryLogTensorDeallocation : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"allocation_id")]
        public long AllocationId { get; set; }

        [global::ProtoBuf.ProtoMember(2, Name = @"allocator_name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string AllocatorName { get; set; } = "";

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class MemoryLogTensorOutput : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"step_id")]
        public long StepId { get; set; }

        [global::ProtoBuf.ProtoMember(2, Name = @"kernel_name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string KernelName { get; set; } = "";

        [global::ProtoBuf.ProtoMember(3, Name = @"index")]
        public int Index { get; set; }

        [global::ProtoBuf.ProtoMember(4, Name = @"tensor")]
        public TensorDescription Tensor { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class MemoryLogRawAllocation : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"step_id")]
        public long StepId { get; set; }

        [global::ProtoBuf.ProtoMember(2, Name = @"operation")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Operation { get; set; } = "";

        [global::ProtoBuf.ProtoMember(3, Name = @"num_bytes")]
        public long NumBytes { get; set; }

        [global::ProtoBuf.ProtoMember(4, Name = @"ptr")]
        public ulong Ptr { get; set; }

        [global::ProtoBuf.ProtoMember(5, Name = @"allocation_id")]
        public long AllocationId { get; set; }

        [global::ProtoBuf.ProtoMember(6, Name = @"allocator_name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string AllocatorName { get; set; } = "";

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class MemoryLogRawDeallocation : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"step_id")]
        public long StepId { get; set; }

        [global::ProtoBuf.ProtoMember(2, Name = @"operation")]
        [global::System.ComponentModel.DefaultValue("")]
        public string Operation { get; set; } = "";

        [global::ProtoBuf.ProtoMember(3, Name = @"allocation_id")]
        public long AllocationId { get; set; }

        [global::ProtoBuf.ProtoMember(4, Name = @"allocator_name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string AllocatorName { get; set; } = "";

        [global::ProtoBuf.ProtoMember(5, Name = @"deferred")]
        public bool Deferred { get; set; }

    }

}

#pragma warning restore CS1591, CS0612, CS3021, IDE1006
