
path_info.opt = paths.concat(path_info.dataFolder, 'data_opt.mat')
opt_db = matio.load(path_info.opt, 'opt')

path_info.bbox = paths.concat(path_info.dataFolder, 'data_bbox.mat')
local bbox_db = matio.load(path_info.bbox, 'bbox_mat')

path_info.dataFolder_train = paths.concat(path_info.dataFolder, 'train');
path_info.dataFolder_valid = paths.concat(path_info.dataFolder, 'valid');
path_info.dataFolder_test = paths.concat(path_info.dataFolder, 'test');


if opt.UniDB == 1 then
  opt.n_train = opt_db.n_train[{1,1}]
elseif opt.UniDB == 0 then -- only CCP DB for training
  opt.n_train = opt_db.n_train_1_ccpDB[{1,1}]
end
opt.n_valid = opt_db.n_valid[{1,1}]
opt.n_test = opt_db.n_test[{1,1}]

-------------------------------------------------------------------
path_img = {}
path_img.raw = {}; path_img.bbox = {}; path_img.mask_fg = {};

path_img.raw.train = {}
path_img.mask_fg.train = {}
path_img.bbox.train = bbox_db.train

path_img.raw.valid = {}
path_img.mask_fg.valid = {}
path_img.bbox.valid = bbox_db.valid

path_img.raw.test = {}
path_img.mask_fg.test = {}
path_img.bbox.test = bbox_db.test

for idx = 1, opt.n_train do
  local file_name_raw = string.format('%d_raw.jpg', idx)
  local file_name_fg = string.format('%d_mask_fg.t7', idx)
  
  local file_path_raw = paths.concat(path_info.dataFolder_train, file_name_raw)
  local file_path_fg = paths.concat(path_info.dataFolder_train, file_name_fg)
  table.insert(path_img.raw.train, file_path_raw)
  table.insert(path_img.mask_fg.train, file_path_fg)
end

for idx = 1, opt.n_valid do
  local file_name_raw = string.format('%d_raw.jpg', idx)
  local file_name_fg = string.format('%d_mask_fg.t7', idx)
  
  local file_path_raw = paths.concat(path_info.dataFolder_valid, file_name_raw)
  local file_path_fg = paths.concat(path_info.dataFolder_valid, file_name_fg)
  table.insert(path_img.raw.valid, file_path_raw)
  table.insert(path_img.mask_fg.valid, file_path_fg)
end

for idx = 1, opt.n_test do
  local file_name_raw = string.format('%d_raw.jpg', idx)
  local file_name_fg = string.format('%d_mask_fg.t7', idx)
  
  local file_path_raw = paths.concat(path_info.dataFolder_test, file_name_raw)
  local file_path_fg = paths.concat(path_info.dataFolder_test, file_name_fg)
  table.insert(path_img.raw.test, file_path_raw)
  table.insert(path_img.mask_fg.test, file_path_fg)
end

