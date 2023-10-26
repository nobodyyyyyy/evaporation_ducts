import Vue from 'vue'
import Router from 'vue-router'
import AppIndex from '../home/AppIndex'
import Login from '../components/Login'
import Home from '../components/Home'
import AnalysisDataView from '../components/dataview/AnalysisDataView.vue'
import OriginDataView from '../components/dataview/OriginDataView.vue'
import DuctPredictView from '../components/duct/DuctPredict.vue'
import DuctCalView from '../components/duct/DuctCal.vue'
import DuctView from '../components/duct/DuctView.vue'
import LossView from '../components/radar/LossView.vue'
import LossP from '../components/radar/LossP.vue'
import Main from '../components/Main.vue'

Vue.use(Router)

export default new Router({
  mode: 'history',
  routes: [
    {
      path: '/login',
      name: 'Login',
      component: Login
    },
    {
      path: '/home',
      name: 'Home',
      component: Home,
      // home页面并不需要被访问
      redirect: '/index',
      children: [
        {
          path: '/main',
          name: 'Main',
          component: Main
        },
        {
          path: '/index',
          name: 'AppIndex',
          component: AppIndex
        },
        {
          path: '/AnalysisDataView',
          name: 'AnalysisDataView',
          component: AnalysisDataView
        },
        {
          path: '/OriginDataView',
          name: 'OriginDataView',
          component: OriginDataView
        },
        {
          path: '/DuctPredictView',
          name: 'DuctPredictView',
          component: DuctPredictView
        },
        {
          path: '/DuctCalView',
          name: 'DuctCalView',
          component: DuctCalView
        },
        {
          path: '/DuctView',
          name: 'DuctView',
          component: DuctView
        },
        {
          path: '/LossView',
          name: 'LossView',
          component: LossView
        },
        {
          path: '/LossP',
          name: 'LossP',
          component: LossP
        }
      ]
    },
    {
      path: '/login',
      name: 'Login',
      component: Login
    }
  ]
})
