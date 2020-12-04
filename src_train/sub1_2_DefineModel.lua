
--------------------------------------------------------
if opt.continueFlag == 0 then

  dofile('src_train/subsub1_2_Model.lua')
  model = func_defineModel_yesSkip(opt.resol)

else

  print('- load model: '..path_info.model)
  model = torch.load(path_info.model)

end

--------------------------------------------------------
crit_L1 = nn.AbsCriterion()
print('- L1 AbsCriterion')

-------------------------------------------
local loss_net, crit_args

if opt.vgg == 1 then
  local t0 = sys.clock()
  
  print('- load vgg: '..path_info.vgg_net)
  loss_net = torch.load(path_info.vgg_net)
  print(string.format('  load time for vgg: %.3f sec ',sys.clock()-t0))
  
  if opt.gpuNum then 
    loss_net:cuda()    
  end
  
  crit_args = {
  cnn = loss_net,
  content_layers = opt_vgg.percep_layers,
  content_weights = opt_vgg.percep_weights,
  loss_type = 'L1'
  }
  
  crit_vgg = nn.PerceptualCriterion(crit_args)
  m_vggPrep= nn.VGGpreprocess()  
else
  print('- NO use of VGG loss')  
end


-------------------------------------------
if opt.gpuNum then
  model:cuda()
  crit_L1:cuda()
  
  if opt.vgg == 1 then
    crit_vgg:cuda()
    m_vggPrep:cuda()
  end
end