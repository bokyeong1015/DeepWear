cmd = torch.CmdLine()


cmd:option('-expNum', 1, 'exp number')
cmd:option('-resol', 1, '1 for 192x64, 2 for 384x128')
cmd:option('-UniDB', 1, '1 for unified DB, 0 for only ccp DB')

cmd:option('-vgg', 1, '0 for L1 only | 1 for L1+VGG loss')
cmd:option('-lamb_L1', 1, 'loss weight for L1 loss')
cmd:option('-lamb_vgg', 2, 'loss weight for perceptual loss')

cmd:option('-aug_train', 1, '1 for using data augmentation, 0 for no aug')


cmd:option('-gpuNum', 1, 'gpu number')
cmd:option('-seed', 1, 'random seed')
cmd:option('-threads', 1, 'number of threads')

cmd:option('-continueFlag', 1, '1 for resuming training, 0 for starting from scratch')
cmd:option('-nEpoch', 15, 'number of epochs (final eph)')

cmd:option('-learningRate', 0.0002, 'learning rate')
cmd:option('-weightDecay', 0.0005, 'weight decay')

cmd:option('-batchSize', 10, 'mini-batch size for training')
cmd:option('-batchSize_eval', 20, 'mini-batch size for evaluation')
cmd:option('-flag_printBatch', 1, '1 for print batch result during training, 0 for otherwise')

cmd:option('-plotFlag_learnCurve', 1, '1 for plotting learning curve, 0 for otherwise')
cmd:option('-plotFlag_recon', 1, '1 for plotting recon example, 0 for otherwise')

cmd:option('-nef', 128, '# encoder filters in first conv layer')
cmd:option('-ngf', 128, '# decoder filters in first conv layer')

cmd:option('-ephSave_all', 1, 'epoch frequency for saving results')
cmd:option('-ephSave_minVal', 1, 'epoch freq. for saving results with best validation')


cmd:text()

opt = cmd:parse(arg)

----------------------------------------------------------------

if opt.vgg == 1 then
  opt_vgg = {}

  if opt.resol == 1 then -- conv 1_1/1_2 after relu
    opt_vgg.percep_layers = '2,4' 
    opt_vgg.percep_weights = '1.0,1.0'
    print('** conv 1_1/1_2 after relu')     

  elseif opt.resol == 2 then -- conv 1_1/1_2/2_1/2_2 after relu
    opt_vgg.percep_layers = '2,4,7,9' 
    opt_vgg.percep_weights = '1.0,1.0,1.0,1.0'     
    print('** conv 1_1/1_2/2_1/2_2 after relu') 

  end

  opt_vgg.percep_layers, opt_vgg.percep_weights =
    t.parse_layers(opt_vgg.percep_layers, opt_vgg.percep_weights)

  print(opt_vgg.percep_layers)
  print(opt_vgg.percep_weights)
end

----------------------------------------------------------------
path_info.save = paths.concat(path_info.resultFolder,
  string.format('exp%d_resol%d_vgg%d_uniDB%d',
    opt.expNum, opt.resol, opt.vgg, opt.UniDB))

print('- save to: '..path_info.save)
if (path.exists(path_info.resultFolder) == false) then path.mkdir(path_info.resultFolder) end
if (path.exists(path_info.save) == false) then path.mkdir(path_info.save) end

path_info.save_plot = paths.concat(path_info.save, 'plot_result')
if (path.exists(path_info.save_plot) == false) then path.mkdir(path_info.save_plot) end


----------------------------------------------------------------
if opt.continueFlag == 0 then
  opt.curEpoch = 0
else
  opt.curEpoch = opt.nEpoch
  path_info.model = paths.concat(path_info.save, 'model_eph'..opt.curEpoch..'.t7')

  while(paths.filep(path_info.model) == false) do
    print('- no model: '..path_info.model)

    opt.curEpoch = opt.curEpoch - 1
    path_info.model = paths.concat(path_info.save, 'model_eph'..opt.curEpoch..'.t7')
  end

  print(string.format('    *** curEpoch = %d', opt.curEpoch))
end

----------------------------------------------------------------
cmd:log(paths.concat(path_info.save,
  string.format('log_screen_%d.txt', opt.curEpoch)), opt)

----------------------------------------------------------------
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

torch.setdefaulttensortype('torch.FloatTensor')

if opt.gpuNum then
  cutorch.setDevice(opt.gpuNum)
end

----------------------------------------------------------------
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay
}

optimFunc = optim.adam
print('- optim method: Adam')

----------------------------------------------------------------
print('- load data: flagUniDB '..opt.UniDB)

dofile 'src_train/subsub1_1_DataLoad.lua'

print('  # samples train '..opt.n_train..
  ' | valid '..opt.n_valid..' | test '..opt.n_test)
  

----------------------------------------------------------------  

if opt.resol == 1 then -- 1 for 192x64
  opt.t_scale = torch.FloatTensor({197, 66}) -- [h,w]
  opt.t_crop = torch.FloatTensor({192, 64}) -- [h,w]

elseif opt.resol == 2 then -- 2 for 384x128
  -- 384/0.975 = 393.8462 ~ 394 | 394*(1-0.025) = 384.15 ~ 384
  -- 128/0.975 = 131.2821 ~ 132 | 132*(1-0.025) = 128.7 ~ 128
  opt.t_scale = torch.FloatTensor({394, 132}) -- [h,w]
  opt.t_crop = torch.FloatTensor({384, 128}) -- [h,w]

end

print('  input resol '..opt.resol..': '..opt.t_crop[1]..' '..opt.t_crop[2])