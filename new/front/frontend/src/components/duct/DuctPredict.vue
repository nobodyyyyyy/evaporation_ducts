<template>
  <div id="wrapper">
    <el-container class="container" style="height: 100%">
      <el-header class="navi_header">
        <el-row>
          <el-col :span="24">
            <el-breadcrumb separator-class="el-icon-arrow-right" class="breadcrumb-wrapper child">
              <!--          <el-breadcrumb-item :to="{ path: '/' }">首页</el-breadcrumb-item>-->
              <el-breadcrumb-item>大气波导</el-breadcrumb-item>
              <el-breadcrumb-item>波导预测</el-breadcrumb-item>
            </el-breadcrumb>
          </el-col>
        </el-row>
        <el-row class="title-row">
          <el-col :span="24"><span class="title-text">大气波导预测模型训练及效果评估</span></el-col>
        </el-row>
        <el-row class="sub-title-row">
          <el-col :span="24"><span class="sub-title-text">目前只针对数据缺值较少、探测点位较低的站点进行了波导预计算</span></el-col>
        </el-row>
      </el-header>
      <el-main>
        <div class="search-box">
          <div class="search-row">
            <span class="search-text">站点选择</span>
            <span class="search-text">编号</span>
            <template>
              <el-select v-model="idxSelected" placeholder="请选择站点编号"
                         no-data-text="" filterable :disabled="idxSelected === '加载中'">
                <el-option
                  style="width: 200px"
                  v-for="(item, idx) in station_ids"
                  :key="idx"
                  :label="item"
                  :value="idx">
                </el-option>
              </el-select>
            </template>
            <span class="search-text">|</span>
            <span class="search-text">位置</span>
            <template>
              <el-select v-model="idxSelected" placeholder="请选择地理位置"
                         no-data-text="" filterable :disabled="idxSelected === '加载中'">
                <el-option
                  style="width: 200px"
                  v-for="(item, idx) in locations"
                  :key="idx"
                  :label="item"
                  :value="idx">
                </el-option>
              </el-select>
            </template>
            <i class="el-icon-info select-icon"></i>
            <span class="hint-text">站点编号和地理位置选其一即可</span>
          </div>
          <div class="search-row">
            <span class="search-text">波导高度计算模型</span>
            <template>
              <el-radio v-model="modelSelect" label="nps">nps</el-radio>
              <el-radio v-model="modelSelect" label="babin">byc</el-radio>
              <el-radio v-model="modelSelect" label="pj">pj</el-radio>
              <el-radio v-model="modelSelect" label="liuli">mgb</el-radio>
            </template>
            <i class="el-icon-info select-icon"></i>
            <span class="hint-text">波导预测的特征值即为所选波导高度计算模型的计算结果</span>
          </div>
          <div class="search-row">
            <span class="search-text">预测模型</span>
            <template>
              <el-select v-model="predictModelSelected" placeholder="请选择">
                <el-option
                  v-for="item in predictModelOptions"
                  :key="item[1]"
                  :label="item[0]"
                  :value="item[1]">
                  <span style="float: left">{{ item[0] }}</span>
                  <span style="float: right; color: #8492a6; font-size: 13px">{{ item[1] }}</span>
                </el-option>
              </el-select>
<!--              <el-cascader v-model="predictModelSelected" placeholder="请选择"-->
<!--                           :options="predictModelOptions" :show-all-levels="false"></el-cascader>-->
            </template>
            <span class="search-text">训练数据时间范围</span>
            <el-date-picker
              type="daterange"
              v-model="dateRangeSelected"
              :picker-options="dateOption"
              value-format="timestamp"
              size="small"
              range-separator="至"
              start-placeholder="开始日期"
              end-placeholder="结束日期"
              :default-value="dateFrom"
              @change="datePickerChangeListener"/>
            <i class="el-icon-info select-icon"></i>
            <span class="hint-text">训练时间需大于一个月</span>
          </div>
          <div class="search-row">
            <span class="search-text">预测时间长度</span>
            <el-select v-model="predictLen" placeholder="请选择预测时间长度" no-data-text="" filterable>
              <el-option
                style="width: 200px"
                v-for="item in predictLenOptions"
                :key="item"
                :label="item"
                :value="item">
              </el-option>
            </el-select>
            <i class="el-icon-info select-icon"></i>
            <span class="hint-text">预测时间长度越长，效果越差，建议选择未来48小时(2天)</span>
          </div>
          <div class="search-row">
            <el-button class="search-btn" round v-on:click="onPredictBtnClicked">训练模型</el-button>
            <el-button style="margin-left: 20px" icon="el-icon-setting" round v-on:click="dialogVisible = true" circle></el-button>

            <!--          <i class="el-icon-info select-icon"></i>-->
            <!--          <span class="hint-text">计算结果不会保存</span>-->
          </div>
        </div>
        <el-dialog title="额外设置" :visible.sync="dialogVisible" width="50%" center>
          <div class="dialog-search-row">
            <span class="search-text">粒子群调优</span>
            <el-switch
              v-model="psoRequire"
              :disabled="predictModelSelected === 'LSTM(RNN)' || predictModelSelected === 'GRU(RNN)'">
            </el-switch>
            <i class="el-icon-info select-icon"></i>
            <span class="hint-text">若开启，则有额外的训练时间，请耐心等待</span>
          </div>
          <div class="dialog-search-row">
            <span class="search-text">训练轮次</span>
            <el-input v-model="epoch" @change="resetEpoch" placeholder="请输入内容" style="width: 100px"></el-input>
            <i class="el-icon-info select-icon"></i>
            <span class="hint-text">仅对RNN方法生效，作为深度学习模型训练时的epoch</span>
          </div>
          <div class="dialog-search-row">
            <span class="search-text">训练窗口长度</span>
            <el-input v-model="windowSize" @change="resetWindowSize" placeholder="请输入内容" style="width: 100px"></el-input>
            <i class="el-icon-info select-icon"></i>
            <span class="hint-text">决定使用前几个时间步去预测后续的时间步</span>
          </div>

          <span slot="footer" class="dialog-footer">
            <span class="hint-text">所有设置在修改后即生效</span>
          </span>
        </el-dialog>
        <div class="display-box">
          <el-row class="title-row" style=" margin-top: 15px;">
            <el-col :span="24"><span class="title-text" style="padding-left: 20px; ">历史计算结果</span></el-col>
          </el-row>
          <template>
            <el-table
              class="result-table" :data="tableData" stripe style="width: 95%">
              <el-table-column prop="date" label="计算时间"></el-table-column>
              <el-table-column prop="station" label="所选站点"></el-table-column>
              <el-table-column prop="height_model" label="波导计算模型"></el-table-column>
              <el-table-column prop="predict_model" label="波导预测模型"></el-table-column>
              <el-table-column prop="pso" label="粒子群调优"></el-table-column>
              <el-table-column prop="range" label="训练数据时间范围"></el-table-column>
              <el-table-column prop="output_len" label="预测时间长度"></el-table-column>
              <el-table-column prop="mae" label="MAE"></el-table-column>
              <el-table-column prop="rmse" label="RMSE"></el-table-column>
              <el-table-column prop="mape" label="MAPE(%)"></el-table-column>
            </el-table>
          </template>
        </div>
      </el-main>
    </el-container>
  </div>
</template>

<script>

export default {
  name: 'DuctPredict',
  components: {},
  data () {
    return {
      station_ids: ['加载中'],
      locations: ['加载中'],
      idxSelected: '加载中',
      // 回包相关
      // _this.lats = resp.data.lats
      lngs: [],
      lats: [],
      dateSelectReadOnly: true,
      dateFrom: '',
      dateTo: '',
      dateSelectPlaceHolder: '请等待',
      // 预测模型
      predictModelSelected: '',
      predictModelOptions: [
        ['支持向量回归', 'SVR'], ['K近邻', 'KNN'], ['决策树', 'DT'], ['随机森林', 'RF'], ['梯度提升回归树', 'GBRT'],
        ['长短期记忆', 'LSTM(RNN)'], ['门控循环单元', 'GRU(RNN)']
      ],
      // predictModelOptions: [
      //   {
      //     value: 'ML',
      //     label: '机器学习方法',
      //     children: [
      //       {value: 'DT', label: '决策树'}
      //     ]
      //   }
      // ],
      psoRequire: false,
      // 波导计算模型
      modelSelect: 'nps',
      // 时间选择部分
      dateRangeSelected: '',
      dateOption: {
        disabledDate: (time) => {
          const _this = this
          return time >= _this.dateTo || time <= _this.dateFrom
        }
      },
      predictLen: '未来48小时(2天)',
      predictLenOptions: ['未来24小时(1天)', '未来48小时(2天)', '未来72小时(3天)', '未来96小时(4天)', '未来120小时(5天)', '未来144小时(6天)',
        '未来168小时(7天)', '未来192小时(8天)', '未来216小时(9天)', '未来240小时(10天)',
        '未来264小时(11天)', '未来288小时(12天)', '未来312小时(13天)', '未来336小时(14天)', '未来360小时(15天)', '未来384小时(16天)',
        '未来408小时(17天)', '未来432小时(18天)', '未来456小时(19天)', '未来480小时(20天)'],
      // 设置界面相关
      dialogVisible: false,
      epoch: 50,
      windowSize: 6,
      // 结果展示
      tableData: []
    }
  },
  mounted () {
    this.initEntries()
  },
  methods: {
    initEntries () {
      const _this = this
      const api = 'height/init_entry'
      this.$axios.get(api).then(resp => {
        if (resp && resp.status === 200) {
          console.log(resp)
          _this.station_ids = resp.data.ids
          _this.lats = resp.data.lats
          _this.lngs = resp.data.lngs
          _this.locations = resp.data.locations
          // for (let i = 0; i < _this.station_ids.length; ++i) {
          //   _this.stationsOptions.push({id: _this.station_ids[i], loc: _this.locations[i]})
          // }
          _this.idxSelected = 0
          _this.dateSelectReadOnly = false
          _this.dateFrom = new Date(resp.data.date_from).getTime()
          _this.dateTo = new Date(resp.data.predict_date_to).getTime() // 预测要砍掉几天，不然无法验证
          _this.dateSelectPlaceHolder = '请选择'
        }
      })
    },
    datePickerChangeListener (val) {
      // 时间选择监听
      const _this = this
      let interval = val[1] - val[0]
      interval = interval / 1000
      if (interval < 2592000) { // 一个月，30天
        this.$notify({
          title: '条件错误',
          message: '请选择一个月以上的训练数据，以达到更好的训练效果',
          type: 'warning'
        })
        _this.dateRangeSelected = ''
      }
    },
    onPredictBtnClicked () {
      const _this = this
      if (_this.modelSelect === '' || _this.dateRangeSelected === '') {
        this.$notify({
          title: '条件未输入完全',
          message: '请填入所有查询条件后重试',
          type: 'warning'
        })
        return
      }
      const api = 'predict/predict'
      const data = {
        id: _this.station_ids[_this.idxSelected],
        date: _this.dateRangeSelected,
        pso: _this.psoRequire,
        predict_model: _this.predictModelSelected,
        height_model: _this.modelSelect,
        epoch: _this.epoch,
        window: _this.windowSize,
        output_len: _this.predictLen
      }
      const loading = this.$loading({
        lock: true,
        text: '模型训练中，请稍后……',
        spinner: 'el-icon-loading',
        background: 'rgba(0,0,0,0.5)'
      })
      this.$axios
        .post(api, data)
        .then(successResponse => {
          const code = successResponse.data.code
          if (code === 0) {
            this.$notify({
              title: '模型训练完毕',
              message: successResponse.data.msg,
              type: 'success'
            })
            console.log(successResponse.data.table_entry)
            _this.tableData.unshift(successResponse.data.table_entry)
          } else if (code === -1) {
            this.$notify({
              title: '结果出错',
              message: successResponse.data.msg,
              type: 'warning'
            })
          }
        })
        .catch(failResponse => {
          _this.loading = false
          console.log(failResponse)
          this.$notify({
            title: '无法计算',
            message: '后端出错',
            type: 'error'
          })
        })
      loading.close()
    },
    resetEpoch () {
      let str = this.epoch
      this.epoch = str.replace(/[^\d]/g, '').replace(/\./g, '')
      this.epoch = Number(this.epoch)
      if (this.epoch > 1000) {
        this.epoch = 1000
      } else if (this.epoch < 20) {
        this.epoch = 20
      }
    },
    resetWindowSize () {
      let str = this.windowSize
      this.windowSize = str.replace(/[^\d]/g, '').replace(/\./g, '')
      this.windowSize = Number(this.windowSize)
      if (this.windowSize > 12) {
        this.windowSize = 12
      } else if (this.windowSize < 2) {
        this.windowSize = 2
      }
    }
  }
}
</script>

<style scoped>
#wrapper {
  height: 95%;
}

.navi_header {
  background-color: white;
  color: #333;
  height: 120px !important;
  border-radius: 10px;
}

.el-main {
  background-color: #E9EEF3;
  color: #333;
  text-align: center;
//line-height: 160px;
}

body > .el-container {
  margin-bottom: 40px;
}

.breadcrumb-wrapper {
  margin-top: 20px;
  width: 100%;
}

.el-col {
  text-align: left;
}

.title-row {
  height: 20px;
  text-align: left;
  margin-top: 8px;
}

.sub-title-row {
  height: 20px;
  text-align: left;
  margin-top: 8px;
}

.title-text {
  text-align: left;
  margin-right: 12px;
  margin-bottom: 0;
  color: rgba(0,0,0,.85);
  font-weight: 600;
  font-size: 20px;
  line-height: 32px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.sub-title-text {
  text-align: left;
  margin-right: 12px;
  margin-bottom: 0;
  color: rgba(79, 76, 76, 0.85);
  font-weight: 400;
  font-size: 16px;
  line-height: 32px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.hint-text {
  color: rgba(98, 97, 97, 0.63);
  text-overflow: ellipsis;
  font-size: 14px;
}

.latlng-hint-text {
  color: rgba(98, 97, 97, 0.63);
  text-overflow: ellipsis;
  font-size: 14px;
  margin-left: 12px;
  margin-right: 12px;
}

.search-box {
  margin-left: -20px;
  margin-right: -20px;
  min-height: 200px;
  background-color: white;
  border-radius: 10px;
}

.display-box {
  margin-left: -20px;
  margin-right: -20px;
  margin-top: 20px;
  min-height: 170px;
  background-color: white;
  border-radius: 10px;
  overflow: auto;
}

.search-row {
  text-align: left;
  padding-top: 20px;
}

.dialog-search-row {
  text-align: left;
  padding-top: 20px;
}

.search-text  {
  color: rgba(0,0,0,.85);
  font-weight: 500;
  font-size: 16px;
  margin-left: 23px;
  margin-right: 23px;
}

.select-icon {
  margin-left: 20px;
  margin-right: 10px;
}
.result-table {
  padding: 20px 0 20px 20px;
}
.search-btn {
  width: 150px;
  margin-left: 20px;
  margin-bottom: 20px;
}

/deep/ .el-dialog {
  border-radius: 15px;
}
</style>
