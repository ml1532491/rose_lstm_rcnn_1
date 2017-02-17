import mxnet as mx
import proposal
import proposal_target
import symbol_vgg
import numpy as np
from collections import namedtuple
from rcnn.config import config
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states", "forward_state", "backward_state",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])
def get_all_vgg_rpn_param():
    sym=symbol_vgg.get_vgg_train()
    arg_names=sym.list_arguments()
    arg_params={}
    for k in arg_names:
        arg_params[k]=mx.symbol.Variable(name=k)
    return arg_params


def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)

    i2h = mx.sym.Convolution(data=indata,
                             workspace=2048,
                             kernel=(1, 1), pad=(0, 0), stride=(1, 1),
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_filter=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx)
                             )

    h2h = mx.sym.Convolution(data=prev_state.h,
                             workspace=2048,
                             kernel=(5, 5), pad=(2, 2), stride=(1, 1),

                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_filter=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)
def get_lstm_vgg_conv(data, param, seqidx):
    """
       shared convolutional layers
       :param data: Symbol
       :return: Symbol
       """
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048, name="conv1_1_%d" % seqidx,weight=param["conv1_1_weight"],
                                bias=param["conv1_1_bias"])
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1_%d" % seqidx)
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, workspace=2048,name="conv1_2_%d" % seqidx,weight=param["conv1_2_weight"],
                                bias=param["conv1_2_bias"])
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2_%d" % seqidx)
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1_%d" % seqidx)
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048,
        name="conv2_1_%d" % seqidx, weight=param["conv2_1_weight"],
        bias=param["conv2_1_bias"]
    )
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1_%d" % seqidx)
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, workspace=2048,
        name="conv2_2_%d" % seqidx, weight=param["conv2_2_weight"],
        bias=param["conv2_2_bias"]
    )
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2_%d" % seqidx)
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2_%d" % seqidx)
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048,
        name="conv3_1_%d" % seqidx, weight=param["conv3_1_weight"],
        bias=param["conv3_1_bias"]
    )
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1_%d" % seqidx)
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048,
        name="conv3_2_%d" % seqidx, weight=param["conv3_2_weight"],
        bias=param["conv3_2_bias"]
    )
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2_%d" % seqidx)
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, workspace=2048,
        name="conv3_3_%d" % seqidx, weight=param["conv3_3_weight"],
        bias=param["conv3_3_bias"]
    )
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3_%d" % seqidx)
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3_%d" % seqidx)
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048,
        name="conv4_1_%d" % seqidx, weight=param["conv4_1_weight"],
        bias=param["conv4_1_bias"]
    )
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1_%d" % seqidx)
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048,
        name="conv4_2_%d" % seqidx, weight=param["conv4_2_weight"],
        bias=param["conv4_2_bias"]
    )
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2_%d" % seqidx)
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048,
        name="conv4_3_%d" % seqidx, weight=param["conv4_3_weight"],
        bias=param["conv4_3_bias"]
    )
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3_%d" % seqidx)
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4_%d" % seqidx)
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048,
        name="conv5_1_%d" % seqidx, weight=param["conv5_1_weight"],
        bias=param["conv5_1_bias"]
    )
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1_%d" % seqidx)
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048,
        name="conv5_2_%d" % seqidx, weight=param["conv5_2_weight"],
        bias=param["conv5_2_bias"]
    )
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2_%d" % seqidx)
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, workspace=2048,
        name="conv5_3_%d" % seqidx, weight=param["conv5_3_weight"],
        bias=param["conv5_3_bias"]
    )
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3_%d" % seqidx)

    return relu5_3
def get_lstm_vgg_train(seq_len, num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS, dropout=0.,num_hidden=1024):
    datas = mx.symbol.Variable(name="data")
    im_infos = mx.symbol.Variable(name="im_info")
    gt_boxess = mx.symbol.Variable(name="gt_boxes")
    rpn_labels = mx.symbol.Variable(name='label')
    rpn_bbox_targets = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weights = mx.symbol.Variable(name='bbox_weight')

    # datas=mx.sym.SliceChannel(data=datas, num_outputs=seq_len, squeeze_axis=1)
    im_infos = mx.sym.SliceChannel(data=im_infos, num_outputs=seq_len, axis=0)
    # im_infos = mx.sym.SliceChannel(data=im_infos[0], num_outputs=20)
    gt_boxess = mx.sym.SliceChannel(data=gt_boxess, num_outputs=seq_len, axis=0)
    rpn_labels = mx.sym.SliceChannel(data=rpn_labels, num_outputs=seq_len, axis=0)
    rpn_bbox_targets = mx.sym.SliceChannel(data=rpn_bbox_targets, num_outputs=seq_len, axis=0)
    rpn_bbox_weights = mx.sym.SliceChannel(data=rpn_bbox_weights, num_outputs=seq_len, axis=0)

    param=get_all_vgg_rpn_param()
    wordvec=get_lstm_vgg_conv(datas,param,0)
    wordvec=mx.sym.SliceChannel(data=wordvec, num_outputs=seq_len, axis=0)
    # wordvec=[]
    # # embed
    # for seqidx in range(seq_len):
    #     wordvec.append(get_lstm_vgg_conv(datas[seqidx],param,seqidx))
    last_states = []
    last_states.append(LSTMState(c=mx.sym.Variable("l0_init_c"), h=mx.sym.Variable("l0_init_h")))
    last_states.append(LSTMState(c=mx.sym.Variable("l1_init_c"), h=mx.sym.Variable("l1_init_h")))
    forward_param = LSTMParam(i2h_weight=mx.sym.Variable("l0_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l0_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l0_h2h_weight"),
                              h2h_bias=mx.sym.Variable("l0_h2h_bias"))


    backward_param = LSTMParam(i2h_weight=mx.sym.Variable("l1_i2h_weight"),
                           i2h_bias=mx.sym.Variable("l1_i2h_bias"),
                           h2h_weight=mx.sym.Variable("l1_h2h_weight"),
                           h2h_bias=mx.sym.Variable("l1_h2h_bias"))


    forward_hidden = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[0],
                          param=forward_param,
                          seqidx=seqidx, layeridx=0, dropout=dropout)
        hidden = next_state.h
        last_states[0] = next_state
        forward_hidden.append(hidden)

    backward_hidden = []
    for seqidx in range(seq_len):
        k = seq_len - seqidx - 1
        hidden = wordvec[k]
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[1],
                          param=backward_param,
                          seqidx=k, layeridx=1, dropout=dropout)
        hidden = next_state.h
        last_states[1] = next_state
        backward_hidden.insert(0, hidden)
    loss=[]


    for seqidx in range(seq_len):
        im_info = im_infos[seqidx]
        gt_boxes = gt_boxess[seqidx]
        rpn_label = rpn_labels[seqidx]
        rpn_bbox_target = rpn_bbox_targets[seqidx]
        rpn_bbox_weight = rpn_bbox_weights[seqidx]

        hidden=mx.sym.Concat(*[forward_hidden[seqidx], backward_hidden[seqidx]], dim=1)
        # RPN layers
        rpn_conv = mx.symbol.Convolution(
            data=hidden, kernel=(3, 3), pad=(1, 1), num_filter=512,name="rpn_conv_3x3_%d" % seqidx, weight=param["rpn_conv_3x3_weight"],
        bias=param["rpn_conv_3x3_bias"])
        rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu_%d"%seqidx)
        rpn_cls_score = mx.symbol.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score_%d" % seqidx, weight=param["rpn_cls_score_weight"],
        bias=param["rpn_cls_score_bias"])
        rpn_bbox_pred = mx.symbol.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred_%d" % seqidx, weight=param["rpn_bbox_pred_weight"],
        bias=param["rpn_bbox_pred_bias"])

        # prepare rpn data
        rpn_cls_score_reshape = mx.symbol.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape_%d"%seqidx)

        # classification
        rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                               normalization='valid', use_ignore=True, ignore_label=-1,
                                               name="rpn_cls_prob_%d"%seqidx)
        # bounding box regression
        rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_%d_'%seqidx, scalar=3.0,
                                                               data=(rpn_bbox_pred - rpn_bbox_target))
        rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss_%d'%seqidx, data=rpn_bbox_loss_,
                                        grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

        # ROI proposal
        rpn_cls_act = mx.symbol.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act_%d"%seqidx)
        rpn_cls_act_reshape = mx.symbol.Reshape(
            data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape_%d'%seqidx)
        if config.TRAIN.CXX_PROPOSAL:
            rois = mx.symbol.Proposal(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois_%d'%seqidx,
                feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES),
                ratios=tuple(config.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)
        else:
            rois = mx.symbol.Custom(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois_%d'%seqidx,
                op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
                scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)

        # ROI proposal target
        gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape_%d'%seqidx)
        group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',name='proposal_target_%d'%seqidx,
                                 num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                                 batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
        rois = group[0]
        label = group[1]
        bbox_target = group[2]
        bbox_weight = group[3]

        # Fast R-CNN
        pool5 = mx.symbol.ROIPooling(
            name='roi_pool5_%d'%seqidx, data=hidden, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
        # group 6
        flatten = mx.symbol.Flatten(data=pool5, name="flatten_%d"%seqidx)
        fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6_%d" % seqidx, weight=param["fc6_weight"],
        bias=param["fc6_bias"])
        relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6_%d"%seqidx)
        drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6_%d"%seqidx)
        # group 7
        fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7_%d" % seqidx, weight=param["fc7_weight"],
        bias=param["fc7_bias"])
        relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7_%d"%seqidx)
        drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7_%d"%seqidx)
        # classification
        cls_score = mx.symbol.FullyConnected(name='cls_score_%d' % seqidx, weight=param["cls_score_weight"],
        bias=param["cls_score_bias"], data=drop7, num_hidden=num_classes)
        cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob_%d'%seqidx, data=cls_score, label=label, normalization='batch')
        # bounding box regression
        bbox_pred = mx.symbol.FullyConnected(name='bbox_pred_%d' % seqidx, weight=param["bbox_pred_weight"],
        bias=param["bbox_pred_bias"], data=drop7, num_hidden=num_classes * 4)
        bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_%d_'%seqidx, scalar=1.0, data=(bbox_pred - bbox_target))
        bbox_loss = mx.sym.MakeLoss(name='bbox_loss_%d'%seqidx, data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

        # reshape output
        label = mx.symbol.Reshape(data=label, shape=(config.TRAIN.BATCH_IMAGES, -1), name='label_reshape_%d'%seqidx)
        cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes),
                                     name='cls_prob_reshape_%d'%seqidx)
        bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes),
                                      name='bbox_loss_reshape_%d'%seqidx)
        loss.append([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.symbol.BlockGrad(label)])
    loss_final=[]
    for i in range(5):
        loss_=[loss[j][i] for j in range(seq_len)]
        loss_final.append(mx.sym.Concat(*loss_,dim=1))
    group = mx.symbol.Group(loss_final)
    return group


def get_lstm_vgg_test(seq_len, num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS, dropout=0.):
    datas = mx.symbol.Variable(name="data")
    im_infos = mx.symbol.Variable(name="im_info")


    # datas = mx.sym.SliceChannel(data=datas, num_outputs=seq_len, squeeze_axis=1)
    im_infos = mx.sym.SliceChannel(data=im_infos, num_outputs=seq_len, squeeze_axis=1)

    param = get_all_vgg_rpn_param()
    wordvec = get_lstm_vgg_conv(datas, param, 0)
    wordvec = mx.sym.SliceChannel(data=wordvec, num_outputs=seq_len, dim=0)
    # wordvec = []
    # # embed
    # for seqidx in range(seq_len):
    #     wordvec.append(get_lstm_vgg_conv(datas[seqidx], param, seqidx))
    num_hidden = 1024
    last_states = []
    last_states.append(LSTMState(c=mx.sym.Variable("l0_init_c"), h=mx.sym.Variable("l0_init_h")))
    last_states.append(LSTMState(c=mx.sym.Variable("l1_init_c"), h=mx.sym.Variable("l1_init_h")))
    forward_param = LSTMParam(i2h_weight=mx.sym.Variable("l0_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l0_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l0_h2h_weight"),
                              h2h_bias=mx.sym.Variable("l0_h2h_bias"))

    backward_param = LSTMParam(i2h_weight=mx.sym.Variable("l1_i2h_weight"),
                               i2h_bias=mx.sym.Variable("l1_i2h_bias"),
                               h2h_weight=mx.sym.Variable("l1_h2h_weight"),
                               h2h_bias=mx.sym.Variable("l1_h2h_bias"))

    forward_hidden = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[0],
                          param=forward_param,
                          seqidx=seqidx, layeridx=0, dropout=dropout)
        hidden = next_state.h
        last_states[0] = next_state
        forward_hidden.append(hidden)

    backward_hidden = []
    for seqidx in range(seq_len):
        k = seq_len - seqidx - 1
        hidden = wordvec[k]
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[1],
                          param=backward_param,
                          seqidx=k, layeridx=1, dropout=dropout)
        hidden = next_state.h
        last_states[1] = next_state
        backward_hidden.insert(0, hidden)
    loss = []
    for seqidx in range(seq_len):
        im_info = im_infos[seqidx]

        hidden = mx.sym.Concat(*[forward_hidden[seqidx], backward_hidden[seqidx]], dim=1)
        # RPN layers
        rpn_conv = mx.symbol.Convolution(
            data=hidden, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3_%d" % seqidx,
            weight=param["rpn_conv_3x3_weight"],
            bias=param["rpn_conv_3x3_bias"])
        rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu_%d" % seqidx)
        rpn_cls_score = mx.symbol.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score_%d" % seqidx,
            weight=param["rpn_cls_score_weight"],
            bias=param["rpn_cls_score_bias"])
        rpn_bbox_pred = mx.symbol.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred_%d" % seqidx,
            weight=param["rpn_bbox_pred_weight"],
            bias=param["rpn_bbox_pred_bias"])

        # prepare rpn data
        rpn_cls_score_reshape = mx.symbol.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape_%d" % seqidx)




        # ROI proposal
        rpn_cls_prob = mx.symbol.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act_%d" % seqidx)
        rpn_cls_prob_reshape = mx.symbol.Reshape(
            data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape_%d' % seqidx)
        if config.TEST.CXX_PROPOSAL:
            rois = mx.symbol.Proposal(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois_%d' % seqidx,
                feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES),
                ratios=tuple(config.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
                threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)
        else:
            rois = mx.symbol.Custom(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois_%d' % seqidx,
                op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
                scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
                threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)

        # Fast R-CNN
        pool5 = mx.symbol.ROIPooling(
            name='roi_pool5_%d' % seqidx, data=hidden, rois=rois, pooled_size=(7, 7), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
        # group 6
        flatten = mx.symbol.Flatten(data=pool5, name="flatten_%d" % seqidx)
        fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6_%d" % seqidx,
                                       weight=param["fc6_weight"],
                                       bias=param["fc6_bias"])
        relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6_%d" % seqidx)
        drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6_%d" % seqidx)
        # group 7
        fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7_%d" % seqidx, weight=param["fc7_weight"],
                                       bias=param["fc7_bias"])
        relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7_%d" % seqidx)
        drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7_%d" % seqidx)
        # classification
        cls_score = mx.symbol.FullyConnected(name='cls_score_%d' % seqidx, weight=param["cls_score_weight"],
                                             bias=param["cls_score_bias"], data=drop7, num_hidden=num_classes)
        cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob_%d' % seqidx, data=cls_score, label=label,
                                           normalization='batch')
        # bounding box regression
        bbox_pred = mx.symbol.FullyConnected(name='bbox_pred_%d' % seqidx, weight=param["bbox_pred_weight"],
                                             bias=param["bbox_pred_bias"], data=drop7, num_hidden=num_classes * 4)

        # reshape output

        cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes),
                                     name='cls_prob_reshape_%d' % seqidx)

        loss.extend([rois, cls_prob, bbox_pred])
    group = mx.symbol.Group(loss)
    return group

