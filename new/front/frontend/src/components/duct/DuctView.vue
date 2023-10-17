<template>
  <div id="wrapper">
    <el-container class="container" style="height: 100%">
      <el-header class="navi_header">
        <el-row>
          <el-col :span="24">
            <el-breadcrumb separator-class="el-icon-arrow-right" class="breadcrumb-wrapper child">
              <!--          <el-breadcrumb-item :to="{ path: '/' }">首页</el-breadcrumb-item>-->
              <el-breadcrumb-item>大气波导</el-breadcrumb-item>
              <el-breadcrumb-item>波导展示</el-breadcrumb-item>
            </el-breadcrumb>
          </el-col>
        </el-row>
        <el-row class="title-row">
          <el-col :span="24"><span class="title-text">历史大气波导数据可视化</span></el-col>
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
            <span class="search-text">显示方式</span>
            <el-radio-group v-model="displayModeSelected" size="medium" @change="resultShown = false">
              <el-radio-button v-for="mode in displayModes" :label="mode" :key="mode"/>
            </el-radio-group>
            <span class="search-text">日期选择</span>
            <el-date-picker
              type="daterange"
              v-model="dateRangeSelected"
              :picker-options="dateOption"
              :readonly=dateSelectReadOnly
              :disabled=dateSelectReadOnly
              :default-value="dateFrom"
              value-format="timestamp"
              size="small"
              range-separator="至"
              start-placeholder="开始日期"
              end-placeholder="结束日期"/>
          </div>
          <div class="search-row">
            <el-button class="search-btn" round v-on:click="searchForData">搜索并显示</el-button>
          </div>
        </div>
        <div class="display-box" v-show=resultShown v-loading="loading">
          <template v-if="displayModeSelected === '表格显示'">
            <el-table stripe class="result-table" :data="tableData" style="width: 95%">
              <el-table-column :label="item.cn" :property="item.eng" v-for="item in tableColNameMap" :key="item.eng" align="center">
                <template slot-scope="scope">
                  <span>{{scope.row[scope.column.property]}}</span>
                </template>
              </el-table-column>
            </el-table>
          </template>
          <div id="ductLineChart" style='width: 1200px; height: 800px; overflow: auto; left: 50px'
               v-if="ductLineChartShown"></div>
        </div>
      </el-main>
    </el-container>
  </div>
</template>

<script>

export default {
  name: 'DuctView',
  components: {},
  data () {
    return {
      station_ids: ['加载中'],
      locations: ['加载中'],
      typeSelected: '',
      idxSelected: '加载中',
      lats: [],
      lngs: [],
      dateSelectReadOnly: true,
      dateFrom: '',
      dateTo: '',
      dateRangeSelected: '',
      dateOption: {
        disabledDate: (time) => {
          const _this = this
          return time > _this.dateTo || time < _this.dateFrom
        }
      },
      displayModeSelected: '表格显示',
      displayModes: ['表格显示', '图表显示'],
      resultShown: false,
      ductLineChartShown: false,
      ductLineChartExist: false,
      tableData: [],
      tableColNameMap: new Map(),
      loading: false
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
          _this.tableColNameMap = resp.data.map
        }
      })
    },
    searchForData () {
      const _this = this
      let complete = true
      if (_this.station_ids.length === 0) {
        this.$message({
          showClose: true,
          message: '后端信息未加载完毕，请重新进入界面',
          type: 'error'
        })
        return
      }
      if (_this.dateRangeSelected === '') {
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

      _this.resultShown = true
      _this.loading = true
      // 包头生成
      let api = ''
      let data = {}
      if (_this.displayModeSelected === '表格显示') {
        api = 'height/fetch-height-raw'
      } else {
        api = 'height/fetch-height-graph'
      }
      data = {
        id: _this.station_ids[_this.idxSelected],
        date: _this.dateRangeSelected
      }
      this.clearDrawing()
      this.$axios
        .post(api, data)
        .then(successResponse => {
          if (_this.displayModeSelected === '表格显示') {
            // console.log(successResponse.data.data)
            _this.tableData = successResponse.data.data
          } else {
            _this.ductLineChartShown = true
            _this.ductLineChartExist = true
            this.$nextTick(() => {
              // console.log(successResponse)
              this.drawDuctChart(successResponse)
            })
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
      // _this.ductLineChartShown = true
    },
    drawDuctChart (successResponse) {
      const types = successResponse.data.types
      const titleText = successResponse.data.title
      const xAxis = successResponse.data.x
      const series = successResponse.data.series
      console.log(series)
      console.log(types)
      let chart = this.$echarts.init(document.getElementById('ductLineChart'))
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
    clearDrawing () {
      const _this = this
      if (_this.ductLineChartExist) {
        let lChart = this.$echarts.init(document.getElementById('ductLineChart'))
        lChart.clear()
      }
      _this.ductLineChartExist = false
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
  min-height: 170px;
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
.search-btn {
  width: 150px;
  margin-left: 20px;
  margin-bottom: 20px;
}

</style>
