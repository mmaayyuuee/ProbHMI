from .PRED import PredictBase, Seq2SeqPredictVelonlyDistribNet
from .FlowMatching_PRED import FlowMatchingPredict_V1, FlowMatchingPredict_V2, FlowMatchingPredict_V3
from .FlowMatching_PRED import FlowMatchingPredict_V4, FlowMatchingPredict_V5
from utils.wrapper import this_is_wrapper

# PREDs
prednet_dict = {    
    "seq2seq_velonly_distrib": Seq2SeqPredictVelonlyDistribNet,
    "seq2seq_velonly_distrib_V2": Seq2SeqPredictVelonlyDistribNet,   
         
    "seq2seq_velonly_distrib_V2setZeros": [this_is_wrapper({"set_zeros": True}), Seq2SeqPredictVelonlyDistribNet],
    "seq2seq_velonly_distrib_V2Dropout": [this_is_wrapper({"dropout_coef": 0.2}), Seq2SeqPredictVelonlyDistribNet],
    "seq2seq_velonly_distrib_V2setZerosDropout": [this_is_wrapper({"set_zeros": True, "dropout_coef": 0.2}), Seq2SeqPredictVelonlyDistribNet],

    "seq2seq_velonly_distrib_V2DecoderOnly": [this_is_wrapper({"decoder_only": True}), Seq2SeqPredictVelonlyDistribNet],
    "seq2seq_velonly_distrib_V2DecoderOnlySetZeros": [this_is_wrapper({"decoder_only": True, "set_zeros": True}), Seq2SeqPredictVelonlyDistribNet],
    "seq2seq_velonly_distrib_V2DecoderOnlyDropout": [this_is_wrapper({"decoder_only": True, "dropout_coef": 0.2}), Seq2SeqPredictVelonlyDistribNet],
    "seq2seq_velonly_distrib_V2DecoderOnlySetZerosDropout": [this_is_wrapper({"decoder_only": True, "set_zeros": True, "dropout_coef": 0.2}), Seq2SeqPredictVelonlyDistribNet],    
    
    "seq2seq_velonly_distrib_V2Dct": [this_is_wrapper({"dct_group": True}), Seq2SeqPredictVelonlyDistribNet],
    "seq2seq_velonly_distrib_V2setZerosDct": [this_is_wrapper({"set_zeros": True, "dct_group": True}), Seq2SeqPredictVelonlyDistribNet],
    "seq2seq_velonly_distrib_V2setZerosDropoutDct": [this_is_wrapper({"set_zeros": True, "dct_group": True, "dropout_coef": 0.2}), Seq2SeqPredictVelonlyDistribNet],
    
    "seq2seq_velonly_distrib_V2DecoderOnlyDct": [this_is_wrapper({"decoder_only": True}), Seq2SeqPredictVelonlyDistribNet],
    "seq2seq_velonly_distrib_V2DecoderOnlySetZerosDropoutDct": [this_is_wrapper({"decoder_only": True, "set_zeros": True, "dropout_coef": 0.2, "dct_group": True}), Seq2SeqPredictVelonlyDistribNet],
    
    "seq2seq_velonly_distrib_V2PoseCond": [this_is_wrapper({"pose_cond": True}), Seq2SeqPredictVelonlyDistribNet],
    "seq2seq_velonly_distrib_V2setZerosPoseCond": [this_is_wrapper({"set_zeros": True, "pose_cond": True}), Seq2SeqPredictVelonlyDistribNet],               
}


flowmatching_prednet_dict = {
    'V1': FlowMatchingPredict_V1,
    'Pose_V1': [this_is_wrapper({"output_pose":True}), FlowMatchingPredict_V1],
    'V1_1': [this_is_wrapper({"copy_last_frame":True}), FlowMatchingPredict_V1],
    'V1_2': [this_is_wrapper({"noise_copy_last_frame":True}), FlowMatchingPredict_V1],
    
    'V2': FlowMatchingPredict_V2, 
    'Pose_V2': [this_is_wrapper({"output_pose":True}), FlowMatchingPredict_V2],
    'Node_Pose_V2': [this_is_wrapper({"output_pose":True, "level":"node"}), FlowMatchingPredict_V2],
    'Node_Noise_Pose_V2': [this_is_wrapper({"output_pose":True, "level":"node", "noise_copy":True}), FlowMatchingPredict_V2],
    'Channel_Pose_V2': [this_is_wrapper({"output_pose":True, "level":"channel"}), FlowMatchingPredict_V2],
    'Channel_Noise_Pose_V2': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_copy":True}), FlowMatchingPredict_V2],
    'Channel_without_Noise_Pose_V2': [this_is_wrapper({"output_pose":True, "level":"channel", "without_noise":True}), FlowMatchingPredict_V2],
    'Channel_without_Noise_Latent_Pose_V2': [this_is_wrapper({"output_pose":True, "level":"channel", "without_noise":True, "without_latent_space":True}), FlowMatchingPredict_V2],
    'Channel_NoiseEvalOnly_Pose_V2': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_copy_only_eval":True}), FlowMatchingPredict_V2],
    'Channel_NoiseOnly_Pose_V2': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_only":True}), FlowMatchingPredict_V2],
    'Channel_Noise_without_Root_Pose_V2': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_copy_without_root":True}), FlowMatchingPredict_V2],
    'Channel_Noise_without_Latent_Pose_V2': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_copy":True, "without_latent_space":True}), FlowMatchingPredict_V2],
    'Channel_Noise_Pose_LogNorm_V2': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_copy":True, "lognorm_sampling": True}), FlowMatchingPredict_V2],
    'Channel_Noise_Pose_RootVel_V2': [this_is_wrapper({"output_pose_execpt_root":True, "level":"channel", "noise_copy":True}), FlowMatchingPredict_V2],
    'Channel_Noise_without_Latent_Pose_RootVel_V2': [this_is_wrapper({"output_pose_execpt_root":True, "level":"channel", "noise_copy":True, "without_latent_space":True}), FlowMatchingPredict_V2],
    'VelChannel_Noise_Pose_V2': [this_is_wrapper({"level":"channel", "noise_copy":True}), FlowMatchingPredict_V2],
    'CondHis_Channel_Noise_Pose_V2': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_copy":True, "only_conditional_history":True}), FlowMatchingPredict_V2],
    'Channel_Noise_PoseVel_V2': [this_is_wrapper({"output_pose_velocity":True, "level":"channel", "noise_copy":True}), FlowMatchingPredict_V2],
    
    'Node_Noise_Pose_V3': [this_is_wrapper({"output_pose":True, "level":"node", "noise_copy":True}), FlowMatchingPredict_V3],
    'Channel_Pose_V3': [this_is_wrapper({"output_pose":True, "level":"channel"}), FlowMatchingPredict_V3],
    'Channel_Noise_Pose_V3': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_copy":True}), FlowMatchingPredict_V3],
    'Channel_Noise_without_Latent_Pose_V3': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_copy":True, "without_latent_space":True}), FlowMatchingPredict_V3],
    'Channel_NoiseOnly_Pose_V3': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_only":True}), FlowMatchingPredict_V3],

    'Channel_Pose_V4': [this_is_wrapper({"output_pose":True, "level":"channel"}), FlowMatchingPredict_V4],
    'Channel_Noise_Pose_V4': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_copy":True}), FlowMatchingPredict_V4],
    'Channel_without_Noise_Pose_V4': [this_is_wrapper({"output_pose":True, "level":"channel", "without_noise":True}), FlowMatchingPredict_V4],
    'CondHis_Channel_without_Noise_Pose_V4': [this_is_wrapper({"output_pose":True, "level":"channel", "without_noise":True, "only_conditional_history":True}), FlowMatchingPredict_V4],
    'Channel_FullNoise_Pose_V4': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_full_copy":True}), FlowMatchingPredict_V4],
    'Channel_Noise_without_Latent_Pose_V4': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_copy":True, "without_latent_space":True}), FlowMatchingPredict_V4],
    'Channel_NoiseOnly_Pose_V4': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_only":True}), FlowMatchingPredict_V4],
    'Channel_NoiseOnly_Pose_EC_V4': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_only":True, "embedding_cond": True}), FlowMatchingPredict_V4],
    'Channel_NoiseOnly_Pose_Pre_V4': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_only":True, "preprocessing_cond": True}), FlowMatchingPredict_V4],
    'Channel_NoiseOnly_Pose_Double_V4': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_only":True, "double_cond": True}), FlowMatchingPredict_V4],
    'CondHis_Channel_NoiseOnly_Pose_V4': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_only":True, "only_conditional_history":True}), FlowMatchingPredict_V4],
    'Channel_NoiseTrueOnly_Pose_V4': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_true_only":True}), FlowMatchingPredict_V4],
    'Channel_NoiseTrueOnly_Pose_EC_V4': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_true_only":True, "embedding_cond": True}), FlowMatchingPredict_V4],
    'Channel_NoiseTrueOnly_Pose_Pre_V4': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_true_only":True, "preprocessing_cond": True}), FlowMatchingPredict_V4],
    'Channel_NoiseTrueOnly_Pose_Double_V4': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_true_only":True, "double_cond": True}), FlowMatchingPredict_V4],
    'Channel_FullNoiseOnly_Pose_V4': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_full_only":True}), FlowMatchingPredict_V4],
    'CondHis_Channel_NoiseOnly_Pose_EC_V4': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_only":True, "embedding_cond": True, "only_conditional_history":True}), FlowMatchingPredict_V4],

    'Channel_Noise_Pose_V5': [this_is_wrapper({"output_pose":True, "level":"channel", "noise_copy":True}), FlowMatchingPredict_V5],
}



def load_prednet(key):
    prednet_model = prednet_dict[key]
    if isinstance(prednet_model, list):
        prednet = prednet_model[0](prednet_model[1])
    else:
        prednet = prednet_model
    return prednet


def load_flowmatching_prednet(key):
    fm_pred_model = flowmatching_prednet_dict[key]
    if isinstance(fm_pred_model, list):
        fm_pred = fm_pred_model[0](fm_pred_model[1])
    else:
        fm_pred = fm_pred_model
    return fm_pred