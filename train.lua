require 'torch'; require 'nn'; require 'nngraph';
require 'cudnn'; require 'cunn'; require 'optim';
require 'image'; require 'gnuplot'

require 'src_train.src_percepLoss.PerceptualCriterion'
require 'src_train.src_percepLoss.ContentLoss'
require 'src_train.src_percepLoss.VGGpreprocess'

matio = require 'matio'

-------------------------------------------------
path_info = {}

path_info.resultFolder = 'result_train'
path_info.dataFolder = 'data/unifiedDB' -- used in src_train/subsub1_1_DataLoad.lua
path_info.vgg_net = 'src_train/src_percepLoss/VGG_ILSVRC_19_layers_nn.t7'

path_info.utils = 'src_train/utils_v2.lua'
t = dofile(path_info.utils) 

-------------------------------------------------
print(' ') print('=====================================')

print('1.1 Setup & LoadData')
dofile 'src_train/sub1_1_Setup_DataLoad.lua'

print('1.2 Define Model')
dofile 'src_train/sub1_2_DefineModel.lua'

print('1.3 Define function')
dofile 'src_train/sub1_3_DefineFunc.lua'

-------------------------------------------------
print(' ') print('=====================================')

print('2 Main')

matio.save(paths.concat(path_info.save, 'opt.mat'), opt)
torch.save(paths.concat(path_info.save, 'opt.t7'), opt)

local X_plot = func_data_exImg(3) -- loading test data for plot

---------------------------------------------------
if opt.continueFlag == 0 then -- Evaluation @ Epoch0

  info = {}
  info.time_all = sys.clock()

  epoch = 0

  local time_eval = sys.clock()
  local eval_valid = func_eval(2)
  time_eval = sys.clock() - time_eval

  func_update_info(epoch, eval_valid, time_eval, 0)
  if opt.plotFlag_recon == 1 then func_plot_exImg(3, epoch, X_plot) end

  model:clearState()
  func_save_epoch(epoch)

else -- Load previous eval info

  print('    - load '..string.format('infoEpoch_eph%d.mat', opt.curEpoch))
  info = torch.load(paths.concat(path_info.save,
    string.format('infoEpoch_eph%d.t7', opt.curEpoch)))

  print('    - load '..string.format('infoBatch_eph%d.mat', opt.curEpoch))
  info_batch = torch.load(paths.concat(path_info.save,
    string.format('infoBatch_eph%d.t7', opt.curEpoch)))

  info.time_all = sys.clock()

end

-------------------------------------------
-- Start Training
for epoch = opt.curEpoch+1, opt.nEpoch do

  -- train
  local time_train = sys.clock()
  func_train(epoch)
  time_train = sys.clock() - time_train

  -- eval
  local time_eval = sys.clock()
  local eval_valid = func_eval(2)
  time_eval = sys.clock() - time_eval

  -- update info
  func_update_info(epoch, eval_valid, time_eval, time_train)
  if opt.plotFlag_recon == 1 then func_plot_exImg(3, epoch, X_plot) end
  if opt.plotFlag_learnCurve == 1 then func_plot_learnCurve(epoch) end

  -- save  
  model:clearState()  
  if epoch % opt.ephSave_minVal == 0 or epoch == opt.nEpoch then
    func_save_onlyMinVal(epoch)
  end
  if epoch % opt.ephSave_all == 0 or epoch == opt.nEpoch then
    func_save_epoch(epoch)
  end

end

-------------------------------------------------------
print(' ') print('=====================================')
print('3 Final evaluation')

dofile 'src_train/sub1_4_FinalEval.lua'