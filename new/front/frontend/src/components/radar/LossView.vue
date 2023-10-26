<template>
  <div id="wrapper">
    <el-container class="container" style="height: 100%">
      <el-header class="navi_header">
        <el-row>
          <el-col :span="24">
            <el-breadcrumb separator-class="el-icon-arrow-right" class="breadcrumb-wrapper child">
              <!--          <el-breadcrumb-item :to="{ path: '/' }">首页</el-breadcrumb-item>-->
              <el-breadcrumb-item>波导模拟</el-breadcrumb-item>
              <el-breadcrumb-item>损耗及盲区检测</el-breadcrumb-item>
            </el-breadcrumb>
          </el-col>
        </el-row>
        <el-row class="title-row">
          <el-col :span="24"><span class="title-text">大气波导传输损耗及盲区检测</span></el-col>
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
            <span class="search-text">时间选择</span>
            <el-date-picker
              v-model="singleDateSelect" type="date"
              :placeholder=dateSelectPlaceHolder
              :readonly=dateSelectReadOnly
              :disabled=dateSelectReadOnly
              size="small"
              :picker-options="dateOption"
              :default-value="dateFrom"
              value-format="timestamp"/>
          </div>
          <div class="search-row">
            <span class="search-text">天线高度&ensp;&ensp;&ensp;&ensp;</span>
            <el-input placeholder="请输入风速" v-model="hgt" @change="resetHgt">
              <template slot="append">m</template>
            </el-input>
            <span class="search-text" style="margin-left: 60px">雷达频率&ensp;&ensp;&ensp;&ensp;</span>
            <el-input placeholder="请输入压强" v-model="freq" @change="resetFreq">
              <template slot="append">MHz</template>
            </el-input>
          </div>
          <div class="search-row">
            <span class="search-text">自定义波导高度</span>
            <el-switch v-model="selfDefineHgt" style="margin-right: 40px"></el-switch>
            <el-input placeholder="请输入自定义波导高度" v-model="ductHeight" @change="resetHgt" :disabled="!selfDefineHgt">
              <template slot="append">m</template>
            </el-input>
            <i class="el-icon-info select-icon"></i>
            <span class="hint-text">若使用自定义波导高度，则忽略所选的站点、模型和时间</span>
          </div>
          <div class="search-row">
            <el-button class="search-btn" round v-on:click="onBtnClicked('loss')">计算损耗</el-button>
            <el-button class="search-btn" round v-on:click="onDetectionBtnClicked">盲区检测</el-button>
          </div>
          <el-dialog title="盲区监测参数设置" :visible.sync="dialogVisible" width="50%" center>
            <div class="dialog-search-row" v-for="item in detectionDisplay" v-bind:key="item[1]">
              <el-row>
                <el-col :span="8">
                  <span class="search-text" style="font-size: 17px">{{item[0]}}</span>
                </el-col>
                <el-col :span="4">
                </el-col>
                <el-col :span="8">
                  <el-input v-model="detectionData[item[1]]" @change="resetHgt" placeholder="请输入内容" style="width: 100%">
                    <template slot="append">{{item[2]}}</template>
                  </el-input>
                </el-col>
                <el-col :span="4">
                </el-col>
              </el-row>
            </div>
            <span slot="footer" class="dialog-footer">
            <el-button class="search-btn" round v-on:click="doBlindDetection">盲区检测</el-button>
          </span>
          </el-dialog>
        </div>
        <div class="display-box" v-show=resultShown v-loading="loading">
          <div id="ductLossHeatMap" style='width: 1200px; height: 800px; overflow: auto; left: 50px' v-if="ductLossHeatMapShown"></div>
        </div>
      </el-main>
    </el-container>
  </div>
</template>

<script>

export default {
  name: 'LossView',
  components: {},
  data () {
    return {
      station_ids: ['加载中'],
      locations: ['加载中'],
      idxSelected: '加载中',
      // 回包相关
      lngs: [],
      lats: [],
      singleDateSelect: '',
      dateSelectReadOnly: true,
      dateFrom: '',
      dateTo: '',
      dateSelectPlaceHolder: '请等待',
      dateRangeSelected: '',
      dateOption: {
        disabledDate: (time) => {
          const _this = this
          return time >= _this.dateTo || time <= _this.dateFrom
        }
      },
      modelSelect: 'nps',
      hgt: 8,
      freq: 6400,
      loading: false,
      // 结果图
      ductLossHeatMapShown: false,
      ductLossHeatMapExist: false,
      resultShown: false,
      dialogVisible: false,
      // 盲区检测相关
      detectionData: {
        pt: 230,
        G: 30,
        D0: 60,
        Bn: 769230,
        Ls: 30,
        F0: 5,
        sigma: 0.2
      },
      detectionDisplay: [
        ['雷达峰值功率', 'pt', 'MHz'],
        ['天线增益', 'G', 'dB'],
        ['最小信噪比', 'D0', 'MHz'],
        ['接收机带宽', 'Bn', 'MHz'],
        ['系统综合损耗', 'Ls', 'dB'],
        ['接收机噪声系数', 'F0', 'dB'],
        ['目标散射截面', 'sigma', 'm*m']
      ],
      // 自定义波导
      selfDefineHgt: false,
      ductHeight: 5
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
          _this.dateTo = new Date(resp.data.date_to).getTime()
          _this.dateSelectPlaceHolder = '请选择'
        }
      })
    },
    onBtnClicked (val) {
      const _this = this
      if ((_this.modelSelect === '' || _this.singleDateSelect === '') && !_this.selfDefineHgt) {
        this.$notify({
          title: '条件未输入完全',
          message: '请填入所有查询条件后重试',
          type: 'warning'
        })
        return
      }
      let data = {
        id: _this.station_ids[_this.idxSelected],
        date: _this.singleDateSelect,
        height_model: _this.modelSelect,
        freq: _this.freq,
        hgt: _this.hgt,
        detection_param: _this.detectionData,
        search_type: val,
        self_define: _this.selfDefineHgt,
        duct_height: _this.ductHeight
      }
      const api = 'radar/loss_cal'
      _this.resultShown = true
      _this.loading = true
      this.$axios
        .post(api, data)
        .then(successResponse => {
          const code = successResponse.data.code
          console.log(successResponse)
          if (code === 0) {
            // console.log(successResponse.data)
            this.$notify({
              title: '计算成功',
              message: successResponse.data.msg,
              type: 'success'
            })
            _this.ductLossHeatMapShown = true
            _this.ductLossHeatMapExist = true
            this.$nextTick(() => {
              this.drawDuctLossHeatMap(successResponse, val)
            })
          } else if (code === -1) {
            this.$notify({
              title: '结果出错',
              message: successResponse.data.msg,
              type: 'warning'
            })
          }
        })
        .catch(failResponse => {
          console.log(failResponse)
          this.$notify({
            title: '无法计算',
            message: '后端出错',
            type: 'error'
          })
        })
      _this.loading = false
    },
    resetFreq () {
      let str = this.freq
      this.freq = str.replace(/[^\d]/g, '').replace(/\./g, '')
      this.freq = Number(this.freq)
      if (this.freq <= 0) {
        this.freq = 1
      }
    },
    resetHgt () {
      let str = this.hgt
      // eslint-disable-next-line no-useless-escape
      this.hgt = str.replace(/[^\.\d]/g, '')
      this.hgt = Number(this.hgt)
      if (this.hgt <= 0) {
        this.hgt = 1
      }
    },
    onDetectionBtnClicked () {
      this.dialogVisible = !this.dialogVisible
    },
    doBlindDetection () {
      this.onBtnClicked('blind')
      this.dialogVisible = false
    },
    drawDuctLossHeatMap (successResponse, type) {
      const data = successResponse.data.data
      const max = successResponse.data.max_value
      const min = successResponse.data.min_value
      const x = successResponse.data.x
      const y = successResponse.data.y
      const titleText = successResponse.data.title
      let hmChart = this.$echarts.init(document.getElementById('ductLossHeatMap'))
      const d = data
        .map(function (item) {
          return [item[1], item[0], item[2] || '-']
        })
      let color = []
      if (type === 'loss') {
        color = ['#0e6cf8', '#f8f3a3', '#ff0000']
      } else {
        color = ['#0e6cf8', '#f34646', '#ff0000']
      }
      // console.log(d)
      hmChart.setOption({
        title: {
          text: titleText,
          left: 'center',
          top: '4%'
        },
        tooltip: {
          position: 'top',
          axisPointer: {
            // 坐标轴指示器，坐标轴触发有效
            type: 'shadow' // 默认为直线，可选为：'line' | 'shadow'
          },
          // formatter: '经度：{b}<br/>纬度：{c[0]}<br/>值：{c1}'
          formatter: function (params) {
            const data = params.data
            if (type === 'loss') {
              return '高度: ' + data[1] + 'm<br/>范围: ' + data[0] + 'km<br/>损耗: ' + data[2]
            } else {
              let blind = '是'
              if (data[2] === 0) {
                blind = '否'
              }
              return '高度: ' + data[1] + 'm<br/>范围: ' + data[0] + 'km<br/>是否盲区: ' + blind
            }
          }
        },
        grid: {
          height: '50%',
          top: '10%'
        },
        xAxis: {
          name: '距离(km)',
          type: 'category',
          data: x,
          splitArea: {
            show: true
          }
        },
        yAxis: {
          name: '高度(m)',
          type: 'category',
          data: y,
          splitArea: {
            show: true
          }
        },
        visualMap: {
          min: min,
          max: max,
          calculable: true,
          orient: 'horizontal',
          left: 'center',
          bottom: '30%',
          precision: 5,
          itemWidth: '20',
          itemHeight: hmChart.getWidth() - 400,
          inRange: {
            color: color // 修改热力图的颜色 淡蓝色=>深蓝色的过度
          }
        },
        series: [
          {
            type: 'heatmap',
            data: d,
            label: {
              show: false
            },
            emphasis: {
              itemStyle: {
                shadowBlur: 10,
                shadowColor: 'rgba(0, 0, 0, 0.5)'
              }
            }
          }
        ],
        toolbox: {
          show: true,
          left: '4%',
          top: '2%',
          feature: {
            saveAsImage: {
              show: true,
              excludeComponents: ['toolbox'],
              pixelRatio: 2
            },
            dataZoom: {
              show: true
            }
          }
        }
      })
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
  padding-top: 5px;
  margin-left: 23px;
  margin-right: 23px;
  height: 100%;
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

.el-input {
  width: 200px;
}

/deep/ .el-dialog {
  border-radius: 15px;
}
</style>
