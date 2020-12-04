
local eval_valid_minLoss, eval_test_minLoss

print(' ') print('******* Final Eval: Min Loss *******')

info = nil
info = torch.load(paths.concat(path_info.save, 'infoEpoch_minValidLoss.t7'))
local eval_epoch_minLoss = info.epoch[-1]

if (paths.filep(paths.concat(path_info.save, 'evalTest_minLoss_eph'..eval_epoch_minLoss..'.t7')) == false) then
  
  print(string.format('=> Load model_minValidLoss.t7 @ eph %d', eval_epoch_minLoss))
  model = nil
  model = torch.load(paths.concat(path_info.save, 'model_minValidLoss.t7'))

  eval_valid_minLoss = func_eval(2)
  print(' ')
  print(string.format('  Valid AllErr %.3f', eval_valid_minLoss.all))
  torch.save(paths.concat(path_info.save, 'evalValid_minLoss_eph'..eval_epoch_minLoss..'.t7'), eval_valid_minLoss)
  matio.save(paths.concat(path_info.save, 'evalValid_minLoss_eph'..eval_epoch_minLoss..'.mat'), eval_valid_minLoss)

  eval_test_minLoss = func_eval(3)
  print(' ')
  print(string.format('  Test AllErr %.3f', eval_test_minLoss.all))
  torch.save(paths.concat(path_info.save, 'evalTest_minLoss_eph'..eval_epoch_minLoss..'.t7'), eval_test_minLoss)
  matio.save(paths.concat(path_info.save, 'evalTest_minLoss_eph'..eval_epoch_minLoss..'.mat'), eval_test_minLoss)
  
else
  print(string.format('No Need to Compute: evalTest_minLoss_eph%d.t7 Exists!',eval_epoch_minLoss))
end
