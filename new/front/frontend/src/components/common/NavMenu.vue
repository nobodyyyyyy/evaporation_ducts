<template>
  <el-menu
    router
    mode="vertical"
    :default-active="'/main'"
    text-color="black"
    >
<!--    <el-menu-item v-for="(item, i) in navList" :key="i" :index="item.name">-->
<!--      {{item.navItem}}-->
<!--    </el-menu-item>-->
    <el-menu-item index="/main" :key="1">首页</el-menu-item>
    <el-submenu index="2">
      <template class="item-entry" slot="title">数据展示</template>
      <el-menu-item index="data-view">探空资料</el-menu-item>
      <el-menu-item index="data-view">再分析资料</el-menu-item>
    </el-submenu>
    <el-menu-item index="/index" :key="3">波导预测</el-menu-item>
    <el-menu-item index="/admin" :key="4">传输损耗</el-menu-item>
<!--    <a href="#nowhere" style="color: #222;float: right;padding: 20px;">更多功能</a>-->
<!--    <i class="el-icon-menu" style="float:right;font-size: 45px;color: #222;padding-top: 8px"></i>-->
<!--    <span style="position: absolute;padding-top: 20px;right: 46%;font-size: 23px;font-weight: bold">短临预测蒸发波导软件</span>-->
  </el-menu>
</template>

<script>
export default {
  name: 'NavMenu',
  data () {
    return {
      navList: [
        {name: '/main', navItem: '首页'},
        {name: '/data-view', navItem: '数据展示'},
        {name: '/index', navItem: '波导预测'},
        {name: '/admin', navItem: '传输损耗'}
      ]
    }
  }
}
</script>

<style scoped>
a{
  text-decoration: none;
}

span {
  pointer-events: none;
}

</style>
