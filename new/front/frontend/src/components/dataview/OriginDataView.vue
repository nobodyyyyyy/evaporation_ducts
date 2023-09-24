<template>
  <div id="wrapper">
    <el-container class="container" style="height: 100%">
      <el-header class="navi_header">
        <el-row>
          <el-col :span="24">
            <el-breadcrumb separator-class="el-icon-arrow-right" class="breadcrumb-wrapper child">
              <!--          <el-breadcrumb-item :to="{ path: '/' }">首页</el-breadcrumb-item>-->
              <el-breadcrumb-item>数据展示</el-breadcrumb-item>
              <el-breadcrumb-item>探空资料</el-breadcrumb-item>
            </el-breadcrumb>
          </el-col>
        </el-row>
        <el-row class="title-row">
          <el-col :span="24"><span class="title-text">探空资料可视化展示</span></el-col>
        </el-row>
      </el-header>
      <el-main>
        <div class="search-box">
          <div class="search-row">
            <span class="search-text">站点选择</span>
            <span class="search-text">编号</span>
            <template>
              <el-select v-model="idxSelected" placeholder="请选择站点编号" @change="stationSelectedChangeEvent"
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
              <el-select v-model="idxSelected" placeholder="请选择地理位置" @change="locationSelectedChangeEvent"
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
            <span class="search-text">显示方式</span>
            <el-radio-group v-model="displayModeSelected" size="medium" @change="onRangeSelectBtnChangeListener">
              <el-radio-button v-for="mode in displayModes" :label="mode" :key="mode"/>
            </el-radio-group>
<!--            日期选择-->
            <span class="search-text">日期选择</span>
            <el-date-picker :class="{'single-date-select': true}"
                            v-model="singleDateSelect" type="date"
                            :placeholder=dateSelectPlaceHolder
                            :readonly=dateSelectReadOnly
                            :disabled=dateSelectReadOnly
                            size="small"
                            :picker-options="dateOption"
                            :default-value="dateFrom"
                            value-format="yyyy-MM-dd"
                            v-show="displayModeSelected === '按天显示'"/>
            <!--            周期显示选择时间段-->
            <el-date-picker
              v-show="displayModeSelected === '按时间段显示'"
              v-model="dateRangeSelected"
              type="daterange"
              :picker-options="dateOption"
              :readonly=dateSelectReadOnly
              :disabled=dateSelectReadOnly
              :default-value="dateFrom"
              value-format="yyyy-MM-dd"
              size="small"
              range-separator="至"
              start-placeholder="开始日期"
              end-placeholder="结束日期"/>
          </div>
          <div class="search-row" v-if="displayModeSelected === '按天显示'">
            <span class="search-text">显示类型</span>
            <el-radio-group v-model="singleDateDisplayModeSelected" size="medium" @change="onSingleDateDisplayModeChangeListener">
              <el-radio-button v-for="mode in singleDateDisplayModes" :label="mode" :key="mode"/>
            </el-radio-group>
            <span class="search-text" v-if="singleDateDisplayModeSelected === '单一数据垂直分布'">单一数据类型</span>
              <el-select v-model="singleDateDisplayTypeSelected" placeholder="请选择数据类型"
                         no-data-text="" filterable :disabled="idxSelected === '加载中'"
                         v-if="singleDateDisplayModeSelected === '单一数据垂直分布'">
                <el-option
                  v-for="item in singleDateColNameMap" :label="item.cn" :value="item.eng" :key="item.eng"
                  style="width: 200px">
                  <span style="float: left">{{ item.cn }}</span>
                  <span style="float: right; color: #8492a6; font-size: 13px;">{{ item.eng }}</span>
                </el-option>
              </el-select>
            <i class="el-icon-info select-icon" v-if="singleDateDisplayModeSelected === '单一数据垂直分布'"></i>
            <span class="hint-text" v-if="singleDateDisplayModeSelected === '单一数据垂直分布'">所选的单一数据类型将作为垂直廓线的x轴</span>
          </div>
          <div class="search-row" v-if="displayModeSelected === '按时间段显示'">
            <span class="search-text">数据类型</span>
            <el-select v-model="typeSelected" placeholder="请选择数据类型" multiple
                       no-data-text="" filterable :disabled="idxSelected === '加载中'">
              <el-option
                v-for="item in colNameMap" :label="item.cn" :value="item.eng" :key="item.eng"
                style="width: 200px">
                <span style="float: left">{{ item.cn }}</span>
                <span style="float: right; color: #8492a6; font-size: 13px; margin-right: -30px">{{ item.eng }}</span>
              </el-option>
            </el-select>
            <i class="el-icon-info select-icon"></i>
            <span class="hint-text">类型选择得越多，数据展示会越杂乱，建议只选择部分需要展示的类型</span>
          </div>
          <div class="search-row" v-if="displayModeSelected === '按时间段显示'">
            <span class="search-text">探测点位</span>
            <template>
              <el-input-number v-model="levelIdx" controls-position="right" :min="1" :max="10" size="median"></el-input-number>
            </template>
            <i class="el-icon-info select-icon"></i>
            <span class="hint-text">由于一天的探空资料在不同高度存在不同探测点，所以需选择要展示趋势的探测点位置</span>
          </div>
          <div class="search-row" v-if="idxSelected > -1">
            <i class="el-icon-success" style="margin-left: 25px"></i>
            <span class="select-text" v-if="displayModeSelected === '按天显示'">
              您选择的地点为 {{locations[idxSelected]}}, 经度 {{lngs[idxSelected]}}°，纬度 {{lats[idxSelected]}}°。日期：{{singleDateSelect}}。
            </span>
            <span class="select-text" v-if="displayModeSelected === '按时间段显示'">
              您选择的地点为 {{locations[idxSelected]}}, 经度 {{lngs[idxSelected]}}°，纬度 {{lats[idxSelected]}}°。
              日期：{{dateRangeSelected[0]}} ~ {{dateRangeSelected[1]}}。
              数据类型：{{typeSelected}}。
              探测点位：{{levelIdx}}。
            </span>
          </div>
          <div class="search-row">
            <el-button class="search-btn" round v-on:click="searchForData">搜索并显示</el-button>
          </div>
        </div>
        <div class="display-box" v-show=resultShown v-loading="loading">
          <template v-if="displayModeSelected === '按天显示' && singleDateDisplayModeSelected === '全部数据'">
            <el-table stripe class="result-table" :data="tableData" style="width: 90%">
              <el-table-column :label="item.cn" :property="item.eng" v-for="item in colNameMap" :key="item.eng" align="center">
                <template slot-scope="scope">
                  <span>{{scope.row[scope.column.property]}}</span>
                </template>
              </el-table-column>
            </el-table>
          </template>
          <div id="stackedLineChart" style='width: 1200px; height: 800px; overflow: auto; left: 50px'
               v-if="displayModeSelected === '按时间段显示'"></div>
          <div id="singleDayLineChart" style='width: 1200px; height: 800px; overflow: auto; left: 50px'
               v-if="displayModeSelected === '按天显示' && singleDateDisplayModeSelected === '单一数据垂直分布'"></div>
        </div>
      </el-main>
    </el-container>
  </div>
</template>

<script>
import SideMenu from './SideMenu.vue'

export default {
  name: 'OriginDataView',
  components: {SideMenu},
  data () {
    return {
      station_ids: ['加载中'],
      locations: ['加载中'],
      typeSelected: '',
      lats: [],
      lngs: [],
      idxSelected: '加载中',
      displayModes: ['按天显示', '按时间段显示'],
      displayModeSelected: '按天显示',
      dateSelectPlaceHolder: '加载中',
      dateSelectReadOnly: true,
      dateFrom: '',
      dateTo: '',
      singleDateSelect: '',
      dateRangeSelected: '',
      dateOption: {
        disabledDate: (time) => {
          const _this = this
          return time > _this.dateTo || time < _this.dateFrom
        }
      },
      // 结果页面
      resultShown: false,
      loading: false,
      tableData: [],
      colNameMap: new Map(),
      tableHeader: [],
      // 周期展示
      levelIdx: 1,
      stackedLineChartExist: false,
      // 廓线、或者是全部显示
      singleDateDisplayModes: ['全部数据', '单一数据垂直分布'],
      singleDateDisplayModeSelected: '全部数据',
      singleDateDisplayTypes: [],
      singleDateDisplayTypeSelected: '',
      singleDateColNameMap: new Map(),
      singleDayLineChartExist: false
    }
  },
  mounted () {
    this.initEntries()
  },
  methods: {
    initEntries () {
      const _this = this
      const api = 'data/origin/init'
      this.$axios.get(api).then(resp => {
        if (resp && resp.status === 200) {
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
    stationSelectedChangeEvent (val) {
    },
    locationSelectedChangeEvent (val) {
    },
    onRangeSelectBtnChangeListener (val) {
      let _this = this
      _this.dateRangeSelected = ''
      _this.singleDateSelect = ''
      console.log(_this.colNameMap)
      if (_this.colNameMap.size === 0) {
        console.log(1)
        const api = 'data/origin/header'
        this.$axios
          .post(api, {})
          .then(successResponse => {
            // console.log(successResponse)
            _this.colNameMap = successResponse.data.map
          })
          .catch(failResponse => {
            _this.loading = false
            console.log(failResponse)
            this.$message({
              showClose: true,
              message: '获取可选数据类型失败',
              type: 'error'
            })
          })
      }
    },
    onSingleDateDisplayModeChangeListener () {
      let _this = this
      if (_this.singleDateColNameMap.size === 0) {
        console.log(1)
        const api = 'data/origin/single-day-entry'
        this.$axios
          .post(api, {})
          .then(successResponse => {
            _this.singleDateColNameMap = successResponse.data.map
          })
          .catch(failResponse => {
            _this.loading = false
            console.log(failResponse)
            this.$message({
              showClose: true,
              message: '获取单一数据类型失败',
              type: 'error'
            })
          })
      }
      _this.clearAllDrawings()
    },
    searchForData () {
      const _this = this

      // 状态检查
      let complete = true
      if (_this.idxSelected === '加载中') {
        complete = false
      }
      if (_this.displayModeSelected === '按天显示' && _this.singleDateSelect === '') {
        complete = false
      }
      if (_this.displayModeSelected === '按时间段显示' && (_this.dateRangeSelected === '' || _this.typeSelected === '')) {
        complete = false
      }
      if (_this.singleDateDisplayModeSelected === '单一数据垂直分布' && _this.singleDateDisplayTypeSelected === '') {
        complete = false
      }
      if (!complete) {
        this.$message({
          showClose: true,
          message: '条件未输入完全',
          type: 'error'
        })
        return
      }

      // 包头生成
      let api = ''
      let data = {}
      _this.loading = true
      _this.resultShown = true
      if (_this.displayModeSelected === '按天显示') {
        if (_this.singleDateDisplayModeSelected === '单一数据垂直分布') {
          api = 'data/origin/fetch-single-daily'
          data = {
            id: _this.station_ids[_this.idxSelected],
            date: _this.singleDateSelect,
            type: _this.singleDateDisplayTypeSelected
          }
        } else {
          // 全部数据，画表格
          api = 'data/origin/fetch-single'
          data = {
            id: _this.station_ids[_this.idxSelected],
            date: _this.singleDateSelect
          }
        }
      } else if (_this.displayModeSelected === '按时间段显示') {
        api = 'data/origin/fetch-range'
        data = {
          id: _this.station_ids[_this.idxSelected],
          date: _this.dateRangeSelected,
          type: _this.typeSelected,
          level: _this.levelIdx
        }
      }

      // 请求
      this.$axios
        .post(api, data)
        .then(successResponse => {
          // console.log(successResponse)
          if (_this.displayModeSelected === '按时间段显示') {
            this.$nextTick(() => {
              this.drawRangeLineChart(successResponse)
              _this.stackedLineChartExist = true
            })
          } else {
            const data = successResponse.data
            if (_this.singleDateDisplayModeSelected === '单一数据垂直分布') {
              this.$nextTick(() => {
                this.drawSingleDayLineChart(successResponse)
                _this.singleDayLineChartExist = true
              })
            } else {
              _this.tableHeader = data.cols_eng
              _this.colNameMap = data.map
              _this.tableData = data.data
            }
          }
          _this.loading = false
        })
        .catch(failResponse => {
          _this.loading = false
          console.log(failResponse)
          this.$message({
            showClose: true,
            message: '后端出错',
            type: 'error'
          })
        })
    },
    drawRangeLineChart (successResponse) {
      const types = successResponse.data.types
      const titleText = successResponse.data.title
      const xAxis = successResponse.data.x
      const series = successResponse.data.series
      console.log(series)
      console.log(types)
      let chart = this.$echarts.init(document.getElementById('stackedLineChart'))
      chart.setOption({
        title: {
          text: titleText,
          left: 'center',
          top: '5%'
        },
        tooltip: {
          trigger: 'axis'
        },
        legend: {
          top: '8%',
          bottom: '5%',
          data: types
        },
        grid: {
          height: '60%',
          left: '3%',
          right: '4%',
          top: '17%',
          containLabel: true
        },
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
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: xAxis
        },
        yAxis: {
          type: 'value'
        },
        series: series
      })
    },
    drawSingleDayLineChart (successResponse) {
      const data = successResponse.data.values
      const heights = successResponse.data.heights
      // const valueMinMax = successResponse.data.value_min_max
      // const heightMinMax = successResponse.data.height_min_max
      const titleText = successResponse.data.title
      const unit = successResponse.data.unit
      const desp = successResponse.data.cn_desp
      let lChart = this.$echarts.init(document.getElementById('singleDayLineChart'))
      console.log(heights)
      console.log(data)
      // 绘制图表
      lChart.setOption({
        title: {
          text: titleText,
          left: 'center',
          top: '5%'
        },
        // legend: {
        //   data: ['Altitude (km) vs. temperature (°C)']
        // },
        tooltip: {
          trigger: 'axis',
          formatter: desp + '<br/>{b}m : {c}' + unit
        },
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
        },
        grid: {
          height: '80%',
          top: '10%',
          containLabel: true
        },
        xAxis: {
          type: 'value',
          axisLabel: {
            formatter: '{value} ' + unit
          }
        },
        yAxis: {
          type: 'category',
          axisLine: { onZero: false },
          axisLabel: {
            formatter: '{value} m'
          },
          boundaryGap: false,
          data: heights
        },
        series: [
          {
            name: titleText,
            type: 'line',
            symbolSize: 10,
            symbol: 'circle',
            smooth: true,
            lineStyle: {
              width: 3,
              shadowColor: 'rgba(0,0,0,0.3)',
              shadowBlur: 10,
              shadowOffsetY: 8
            },
            data: data
          }
        ]
      })
    },
    clearAllDrawings () {
      const _this = this
      if (_this.stackedLineChartExist) {
        let lChart = this.$echarts.init(document.getElementById('stackedLineChart'))
        lChart.clear()
      }
      if (_this.singleDayLineChartExist) {
        let lChart = this.$echarts.init(document.getElementById('singleDayLineChart'))
        lChart.clear()
      }
      _this.stackedLineChartExist = false
      _this.singleDayLineChartExist = false
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
  height: 90px !important;
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

.hint-text {
  color: rgba(98, 97, 97, 0.63);
  text-overflow: ellipsis;
  font-size: 14px;
}

.select-text {
  color: rgba(98, 97, 97, 0.78);
  text-overflow: ellipsis;
  font-size: 16px;
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
  min-height: 170px;
  background-color: white;
  border-radius: 10px;
}

.search-btn {
  margin-left: 20px;
  margin-bottom: 20px;
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
  padding: 20px 0px 20px 20px;
}

// 不显示滚动条
::-webkit-scrollbar {
  width: 0 !important;
}
::-webkit-scrollbar {
  width: 0 !important; height: 0;
}
</style>
