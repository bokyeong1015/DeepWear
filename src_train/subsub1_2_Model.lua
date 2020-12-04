local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') then
    m.weight:normal(0.0, 0.02)
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end



function func_defineModel_yesSkip(input_resol)

  local model

  local input_mask = - nn.Identity()
  local input_fg = - nn.Identity()


  if input_resol == 1 then -- 1 for 192x64

    local input_cat = { input_mask, input_fg }
      - nn.JoinTable(2)
    -- 4 x 192 x 64

    local e1 = input_cat - cudnn.SpatialConvolution(3+1, opt.nef, 4, 4, 2, 2, 1, 1)
    -- state size: (opt.nef) x 96 x 32

    local e2 = e1 - nn.LeakyReLU(0.2, true)
      - cudnn.SpatialConvolution(opt.nef, opt.nef*2, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.nef*2)
    -- state size: (opt.nef*2) x 48 x 16

    local e3 = e2 - nn.LeakyReLU(0.2, true)
      - cudnn.SpatialConvolution(opt.nef* 2, opt.nef * 4, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.nef * 4)
    -- state size: (opt.nef*4) x 24 x 8

    local e4 = e3 - nn.LeakyReLU(0.2, true)
      - cudnn.SpatialConvolution(opt.nef * 4, opt.nef * 8, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.nef * 8)
    -- state size: (opt.nef*8) x 12 x 4

    local e5 = e4 - nn.LeakyReLU(0.2, true)
      - cudnn.SpatialConvolution(opt.nef * 8, opt.nef * 8, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.nef * 8)
    -- state size: (opt.nef*8) x 6 x 2

    local e6 = e5 - nn.LeakyReLU(0.2, true)
      - cudnn.SpatialConvolution(opt.nef * 8, opt.nef * 8, 2, 6, 1, 1, 0, 0)
    -- state size: (opt.nef * 8) x 1 x 1

    local d0_pre = e6 - nn.ReLU(true)
      - nn.SpatialFullConvolution(opt.nef * 8, opt.ngf * 8, 2, 6, 1, 1, 0, 0)
      - nn.SpatialBatchNormalization(opt.ngf * 8)
    -- state size: (opt.ngf*8) x 6 x 2

    local d0 = {d0_pre, e5} - nn.JoinTable(2)
    -- state size: (opt.ngf*8*2) x 6 x 2

    local d1_pre = d0 - nn.ReLU(true)
      - nn.SpatialFullConvolution(opt.nef * 8 * 2, opt.ngf * 8, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.ngf * 8)
    -- state size: (opt.ngf*8) x 12 x 4

    local d1 = {d1_pre, e4} - nn.JoinTable(2)
    -- state size: (opt.ngf*8*2) x 12 x 4

    local d2_pre = d1 - nn.ReLU(true)
      - nn.SpatialFullConvolution(opt.ngf * 8 * 2, opt.ngf * 4, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.ngf * 4)
    -- state size: (opt.ngf*4) x 24 x 8

    local d2 = {d2_pre, e3} - nn.JoinTable(2)
    -- state size: (opt.ngf*4*2) x 24 x 8

    local d3_pre = d2 - nn.ReLU(true)
      - nn.SpatialFullConvolution(opt.ngf * 4 * 2, opt.ngf * 2, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.ngf * 2)
    -- state size: (opt.ngf*2) x 48 x 16

    local d3 = {d3_pre, e2} - nn.JoinTable(2)
    -- state size: (opt.ngf*2*2) x 48 x 16

    local d4_pre = d3 - nn.ReLU(true)
      - nn.SpatialFullConvolution(opt.ngf * 2 * 2, opt.ngf, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.ngf)
    -- state size: (opt.ngf) x 96 x 32

    local d4 = d4_pre - nn.ReLU(true)
      - nn.SpatialFullConvolution(opt.ngf, opt.ngf, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.ngf)
    -- state size: (opt.ngf) x 192 x 64

    local d_out = d4 - nn.ReLU(true)
      - cudnn.SpatialConvolution(opt.ngf, 3, 5, 5, 1, 1, 2, 2)
      - nn.Tanh()
    -- state size: 3 x 192 x 64

    --------------------------------------------------------
    model = nn.gModule({input_mask, input_fg}, {d_out})

    --------------------------------------------------------
    --------------------------------------------------------
  elseif input_resol == 2 then -- 2 for 384x128

    local input_cat = { input_mask, input_fg }
      - nn.JoinTable(2)
    -- 4x384x128

    local e1 = input_cat - cudnn.SpatialConvolution(3+1, opt.nef, 4, 4, 2, 2, 1, 1)
    -- state size: (opt.nef) x 192x64

    local e2 = e1 - nn.LeakyReLU(0.2, true)
      - cudnn.SpatialConvolution(opt.nef, opt.nef*2, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.nef*2)
    -- state size: (opt.nef*2) x 96x32

    local e3 = e2 - nn.LeakyReLU(0.2, true)
      - cudnn.SpatialConvolution(opt.nef* 2, opt.nef * 4, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.nef * 4)
    -- state size: (opt.nef*4) x 48x16

    local e4 = e3 - nn.LeakyReLU(0.2, true)
      - cudnn.SpatialConvolution(opt.nef * 4, opt.nef * 8, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.nef * 8)
    -- state size: (opt.nef*8) x 24x8

    local e5 = e4 - nn.LeakyReLU(0.2, true)
      - cudnn.SpatialConvolution(opt.nef * 8, opt.nef * 8, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.nef * 8)
    -- state size: (opt.nef*8) x 12x4

    local e6 = e5 - nn.LeakyReLU(0.2, true)
      - cudnn.SpatialConvolution(opt.nef * 8, opt.nef * 8, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.nef * 8)
    -- state size: (opt.nef*8) x 6x2

    local e7 = e6 - nn.LeakyReLU(0.2, true)
      - cudnn.SpatialConvolution(opt.nef * 8, opt.nef * 8, 2, 6, 1, 1, 0, 0)
    -- state size: (opt.nef * 8) x 1 x 1

    local d0_pre = e7 - nn.ReLU(true)
      - nn.SpatialFullConvolution(opt.nef * 8, opt.ngf * 8, 2, 6, 1, 1, 0, 0)
      - nn.SpatialBatchNormalization(opt.ngf * 8)
    -- state size: (opt.ngf*8) x 6 x 2

    local d0 = {d0_pre, e6} - nn.JoinTable(2)
    -- state size: (opt.ngf*8*2) x 6 x 2

    local d1_pre = d0 - nn.ReLU(true)
      - nn.SpatialFullConvolution(opt.nef * 8 * 2, opt.ngf * 8, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.ngf * 8)
    -- state size: (opt.ngf*8) x 12 x 4

    local d1 = {d1_pre, e5} - nn.JoinTable(2)
    -- state size: (opt.ngf*8*2) x 12 x 4

    local d2_pre = d1 - nn.ReLU(true)
      - nn.SpatialFullConvolution(opt.ngf * 8 * 2, opt.ngf * 8, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.ngf * 8)
    -- state size: (opt.ngf*8) x 24 x 8

    local d2 = {d2_pre, e4} - nn.JoinTable(2)
    -- state size: (opt.ngf*8*2) x 24 x 8

    local d3_pre = d2 - nn.ReLU(true)
      - nn.SpatialFullConvolution(opt.ngf * 8 * 2, opt.ngf * 4, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.ngf * 4)
    -- state size: (opt.ngf*4) x 48 x 16

    local d3 = {d3_pre, e3} - nn.JoinTable(2)
    -- state size: (opt.ngf*4*2) x 48 x 16

    local d4_pre = d3 - nn.ReLU(true)
      - nn.SpatialFullConvolution(opt.ngf * 4 * 2, opt.ngf * 2, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.ngf * 2)
    -- state size: (opt.ngf*2) x 96 x 32

     local d4 = {d4_pre, e2} - nn.JoinTable(2)
    -- state size: (opt.ngf*2*2) x 96 x 32

    local d5_pre = d4 - nn.ReLU(true)
      - nn.SpatialFullConvolution(opt.ngf * 2 * 2, opt.ngf, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.ngf)
    -- state size: (opt.ngf) x 192 x 64

    local d5 = d5_pre - nn.ReLU(true)
      - nn.SpatialFullConvolution(opt.ngf, opt.ngf, 4, 4, 2, 2, 1, 1)
      - nn.SpatialBatchNormalization(opt.ngf)
    -- state size: (opt.ngf) x 384 x 128

    local d_out = d5 - nn.ReLU(true)
      - cudnn.SpatialConvolution(opt.ngf, 3, 5, 5, 1, 1, 2, 2)
      - nn.Tanh()
    -- state size: 3 x 384 x 128

    --------------------------------------------------------
    model = nn.gModule({input_mask, input_fg}, {d_out})


  end -- end for if input_resol

  --------------------------------------------------------
  --------------------------------------------------------

  model:apply(weights_init)
  graph.dot(model.fg, 'model', paths.concat(path_info.save, 'model_graph'))

  return model

end




