// This file was generated by a tool; you should avoid making direct changes.
// Consider using 'partial classes' to extend these types
// Input: variable.proto

#pragma warning disable CS1591, CS0612, CS3021, IDE1006
namespace Tensorflow
{

    [global::ProtoBuf.ProtoContract()]
    public partial class VariableDef : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"variable_name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string VariableName { get; set; } = "";

        [global::ProtoBuf.ProtoMember(6, Name = @"initial_value_name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string InitialValueName { get; set; } = "";

        [global::ProtoBuf.ProtoMember(2, Name = @"initializer_name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string InitializerName { get; set; } = "";

        [global::ProtoBuf.ProtoMember(3, Name = @"snapshot_name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string SnapshotName { get; set; } = "";

        [global::ProtoBuf.ProtoMember(4, Name = @"save_slice_info_def")]
        public SaveSliceInfoDef SaveSliceInfoDef { get; set; }

        [global::ProtoBuf.ProtoMember(5, Name = @"is_resource")]
        public bool IsResource { get; set; }

        [global::ProtoBuf.ProtoMember(7, Name = @"trainable")]
        public bool Trainable { get; set; }

        [global::ProtoBuf.ProtoMember(8, Name = @"synchronization")]
        public VariableSynchronization Synchronization { get; set; }

        [global::ProtoBuf.ProtoMember(9, Name = @"aggregation")]
        public VariableAggregation Aggregation { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public partial class SaveSliceInfoDef : global::ProtoBuf.IExtensible
    {
        private global::ProtoBuf.IExtension __pbn__extensionData;
        global::ProtoBuf.IExtension global::ProtoBuf.IExtensible.GetExtensionObject(bool createIfMissing)
            => global::ProtoBuf.Extensible.GetExtensionObject(ref __pbn__extensionData, createIfMissing);

        [global::ProtoBuf.ProtoMember(1, Name = @"full_name")]
        [global::System.ComponentModel.DefaultValue("")]
        public string FullName { get; set; } = "";

        [global::ProtoBuf.ProtoMember(2, Name = @"full_shape", IsPacked = true)]
        public long[] FullShapes { get; set; }

        [global::ProtoBuf.ProtoMember(3, Name = @"var_offset", IsPacked = true)]
        public long[] VarOffsets { get; set; }

        [global::ProtoBuf.ProtoMember(4, Name = @"var_shape", IsPacked = true)]
        public long[] VarShapes { get; set; }

    }

    [global::ProtoBuf.ProtoContract()]
    public enum VariableSynchronization
    {
        [global::ProtoBuf.ProtoEnum(Name = @"VARIABLE_SYNCHRONIZATION_AUTO")]
        VariableSynchronizationAuto = 0,
        [global::ProtoBuf.ProtoEnum(Name = @"VARIABLE_SYNCHRONIZATION_NONE")]
        VariableSynchronizationNone = 1,
        [global::ProtoBuf.ProtoEnum(Name = @"VARIABLE_SYNCHRONIZATION_ON_WRITE")]
        VariableSynchronizationOnWrite = 2,
        [global::ProtoBuf.ProtoEnum(Name = @"VARIABLE_SYNCHRONIZATION_ON_READ")]
        VariableSynchronizationOnRead = 3,
    }

    [global::ProtoBuf.ProtoContract()]
    public enum VariableAggregation
    {
        [global::ProtoBuf.ProtoEnum(Name = @"VARIABLE_AGGREGATION_NONE")]
        VariableAggregationNone = 0,
        [global::ProtoBuf.ProtoEnum(Name = @"VARIABLE_AGGREGATION_SUM")]
        VariableAggregationSum = 1,
        [global::ProtoBuf.ProtoEnum(Name = @"VARIABLE_AGGREGATION_MEAN")]
        VariableAggregationMean = 2,
        [global::ProtoBuf.ProtoEnum(Name = @"VARIABLE_AGGREGATION_ONLY_FIRST_REPLICA")]
        VariableAggregationOnlyFirstReplica = 3,
    }

}

#pragma warning restore CS1591, CS0612, CS3021, IDE1006
