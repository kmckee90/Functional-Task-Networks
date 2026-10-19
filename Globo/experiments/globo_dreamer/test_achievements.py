"""Pod-only event integration and persistence tests, without a running game."""
import json
from pathlib import Path
import tempfile
import unittest
import os
import sys
sys.path.insert(0,str(Path(os.environ.get('GLOBO_STATE','/home/dev/globo_dreamer_state'))/'vendor/dreamerv3'))
from .achievements import Achievements
from .achievement_charts import achievement_charts


class AchievementTests(unittest.TestCase):
    def test_chart_matches_checkboxes_not_repetitions(self):
        from .achievement_charts import checked_achievements,achievement_checklist
        row=dict(achievement_counts={'lay_egg':100,'new_food':2},
                 achievement_discoveries={'new_food/potato':10,'new_food/papaya':20},
                 achievement_start_step=1)
        self.assertEqual(checked_achievements(row),4)
        self.assertEqual(achievement_checklist(row).count('☑'),4)
        row['achievement_counts']['lay_egg']=1000
        self.assertEqual(checked_achievements(row),4)

    def test_discoveries_survive_resume_without_duplicates(self):
        with tempfile.TemporaryDirectory() as root:
            a=Achievements(root)
            data=dict(counts={'eat_food':2,'kill_other':1},details={
                'eat_food/main:potatoes':2,'kill_other/animals:gazelle':1,
                'visit_biome/forest':1,'dig_block/soil':1})
            x=a.observe(data,'one',100,1)
            for key in ('new_food','new_species','new_biome','new_resource'):
                self.assertEqual(x['achievement_counts'][key],1)
            b=Achievements(root)
            x=b.observe(data,'two',200,2)
            self.assertEqual(x['achievement_counts']['new_food'],1)
            self.assertEqual(x['achievement_counts']['eat_food'],4)
            from .achievement_charts import run_charts,achievement_checklist
            page=run_charts([dict(x,step=200,total_cells=50)])+achievement_checklist(x)
            self.assertEqual(page.count('Environment observations'),2)
            self.assertIn('Discovered cells',page)
            self.assertIn('<details class="achievement-checklist">',page)

    def test_lives_resume_and_uninstrumented_charts(self):
        with tempfile.TemporaryDirectory() as root:
            a=Achievements(root)
            a.observe(dict(counts={'lay_egg':1}), 'engine1',10,1,initial=True)
            x=a.observe(dict(counts={'lay_egg':2}), 'engine1',11,1)
            self.assertEqual(x['achievement_total'],1)
            a=Achievements(root)
            self.assertEqual(a.observe(dict(counts={'lay_egg':2}),'engine1',12,1)['achievement_delta'],{})
            a.observe(dict(counts={}), 'engine2',13,2,initial=True)
            x=a.observe(dict(counts={'kill_wolf':1,'lay_egg':1}), 'engine2',14,2)
            self.assertEqual(x['achievement_total'],3)
            self.assertEqual(x['achievement_distinct'],2)
            self.assertEqual(x['achievement_first'],{'lay_egg':11,'kill_wolf':14})
            p=Path(root)/'runs/instrumented';p.mkdir(parents=True)
            (p/'status.json').write_text(json.dumps(dict(step=14,environment=x)))
            h=achievement_charts([(19,[dict(name='instrumented'),dict(name='old')])],Path(root),lambda t:t['name'])
            self.assertEqual(h.count('class="achievement-line"'),1)
            self.assertIn('Unavailable: old',h)

    def test_lua_success_attribution_and_return_values(self):
        from lupa import LuaRuntime
        lua=LuaRuntime()
        lua.execute('''
        callbacks={};loaded={};last={};unpack=table.unpack or unpack;
        minetest={registered_nodes={},registered_entities={}}
        minetest.registered_items=minetest.registered_nodes
        function minetest.get_worldpath() return '/test' end
        function minetest.write_json(x) last=x return 'json' end
        function minetest.safe_file_write(...) return true end
        function minetest.register_on_dignode(f) callbacks.dig=f end
        function minetest.register_on_placenode(f) callbacks.place=f end
        function minetest.register_on_mods_loaded(f) table.insert(loaded,f) end
        function minetest.register_globalstep(f) callbacks.step=f end
        function minetest.register_on_player_hpchange(f) callbacks.hp=f end
        function minetest.override_item(n,d) minetest.registered_nodes[n]=d end
        meta={thirst=500};function meta:get_int(k) return self[k] end
        player={};function player:is_player() return true end
        function player:get_meta() return meta end
        animal={};function animal:is_player() return false end
        globo={lay_egg=function(p) return p.success end}
        animals={on_punch=function(self,caps,puncher) self.hp=self.hp-caps.damage return 123 end}
        minetest.registered_nodes['nodes_nature:freshwater_source']={on_rightclick=function(p,n,u) meta.thirst=1000 return 456 end}
        minetest.registered_entities['nodes_nature:drop_water']={on_punch=function(s,p) meta.thirst=math.min(1000,meta.thirst+5) end}
        ''')
        lua.execute(Path(__file__).with_name('achievements.lua').read_text())
        lua.execute('''
        for _,f in ipairs(loaded) do f() end
        player.success=false;assert(globo.lay_egg(player)==false);assert(last.counts.lay_egg==nil)
        player.success=true;assert(globo.lay_egg(player)==true);assert(last.counts.lay_egg==1)
        callbacks.dig({}, {name='dirt'},player);callbacks.place({}, {name='stone'},player)
        assert(last.counts.dig_block==1 and last.details['dig_block/dirt']==1)
        wolf={hp=10,name='animals:wolf'}
        assert(animals.on_punch(wolf,{damage=3},player)==123);assert(last.counts.kill_wolf==nil)
        animals.on_punch(wolf,{damage=8},player);animals.on_punch(wolf,{damage=8},player)
        assert(last.counts.kill_wolf==1)
        animals.on_punch({hp=1,name='animals:cow'},{damage=2},animal);assert(last.counts.kill_cow==nil)
        animals.on_punch({hp=1,name='animals:cow_male'},{damage=2},player);assert(last.counts.kill_cow==1)
        drink=minetest.registered_nodes['nodes_nature:freshwater_source'].on_rightclick
        assert(drink({}, {}, player)==456);drink({}, {}, player);assert(last.counts.drink_water==1)
        meta.thirst=500;minetest.registered_entities['nodes_nature:drop_water'].on_punch({},player)
        assert(last.counts.drink_water==2 and last.details['drink_water/water_drop']==1)
        pos={x=0,y=0,z=0};wet=false
        function player:get_player_name() return 'probe' end
        function player:get_pos() return pos end
        function player:get_hp() return 20 end
        function player:get_inventory() return {get_list=function() return {} end} end
        function meta:get_float(k) return self[k] or 0 end
        function minetest.get_connected_players() return {player} end
        function minetest.get_node(p) return {name=wet and 'water' or 'air'} end
        function minetest.get_item_group(n,g) return n=='water' and 1 or 0 end
        function minetest.get_biome_data(p) return {biome=1} end
        function minetest.get_biome_name(n) return 'forest' end
        callbacks.step(1);pos.y=11;callbacks.step(1);callbacks.step(1)
        assert(last.counts.climb_5m==1 and last.counts.climb_10m==1)
        assert(last.counts.climb_20m==nil)
        wet=true;callbacks.step(1);pos.x=6;wet=false;callbacks.step(1)
        assert(last.counts.cross_water==1)
        callbacks.hp(player,-1,{object={get_luaentity=function() return {name='animals:wolf'} end}})
        callbacks.step(29);assert(last.counts.survive_predator==nil)
        callbacks.step(1);callbacks.step(31);assert(last.counts.survive_predator==1)
        meta.sleeping=1;meta.energy=50;callbacks.step(1);callbacks.step(1)
        assert(last.counts.rest_energy==1 and last.counts.rest_health==nil)
        ''')


if __name__=='__main__':unittest.main()
