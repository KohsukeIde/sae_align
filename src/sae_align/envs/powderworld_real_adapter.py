"""Optional low-level adapter for the original ``kvfrans/powderworld`` package.

Stage D0 uses this adapter only for oracle-positive detector audits.  It avoids
full RL/PSP/Dreamer complexity and exposes the same counterfactual data contract
as the ToyPowderWorld stages: sample an element-ID grid, apply a local
intervention, run no-op vs do-action rollouts, and render observation channels.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Mapping, Tuple
import numpy as np

@dataclass(frozen=True)
class PWAction:
    action_type: str
    element: int
    x: int
    y: int
    radius: int = 2
    def as_array(self) -> np.ndarray:
        return np.array([self.element, self.x, self.y, self.radius], dtype=np.int16)

@dataclass
class PWRollout:
    final_grid: np.ndarray
    event_vector: np.ndarray

class RealPowderworldAdapter:
    # D0 event targets intentionally omit direct "any cell changed" and broad
    # solid/fluid-change indicators. Those are too close to the intervention
    # itself and make the event-present target nearly saturated. The audit needs
    # target events that reflect material-specific action effects.
    EVENT_NAMES = ("fire_gain","fire_loss","water_gain","water_loss","wood_gain","wood_loss","lava_gain","acid_gain","plant_gain")
    def __init__(self, world_size:int=64, seed:int=0, device:str='cpu', use_jit:bool=False):
        try:
            from powderworld.sim import PWSim, PWRenderer
            import torch
        except Exception as exc:  # pragma: no cover
            raise ImportError("RealPowderworldAdapter requires optional `powderworld`. Install via `pip install powderworld` or clone kvfrans/powderworld and `pip install -e .`.") from exc
        self.world_size=int(world_size); self.seed=int(seed); self.rng=np.random.default_rng(self.seed)
        self.device = device
        self.torch = torch
        self.pw=PWSim(device, use_jit=use_jit); self.pwr=PWRenderer(device)
        self.pw_elements:Mapping[str,Tuple[int,int,int]]=self.pw.elements
        self.element_names={int(v[0]):k for k,v in self.pw_elements.items()}; self.element_ids={k:int(v[0]) for k,v in self.pw_elements.items()}
        preferred=['empty','sand','water','fire','wood','stone','wall','lava','acid','plant','dust']
        self.action_elements=[self.element_ids[e] for e in preferred if e in self.element_ids]
        self._next_sim_seed: int | None = None

    def set_rollout_seed(self, seed:int|None)->None:
        self._next_sim_seed = None if seed is None else int(seed)

    def grid_to_world(self, grid:np.ndarray):
        elems = self.torch.as_tensor(np.asarray(grid, dtype=np.int64)[None], dtype=self.torch.long, device=self.device)
        world = self.pw.elem_vecs_array(elems).permute(0, 3, 1, 2).contiguous()
        return world

    def world_to_grid(self, world)->np.ndarray:
        return world[:, : self.pw.NUM_ELEMENTS].argmax(dim=1).detach().cpu().numpy().astype(np.uint8)[0]

    def sample_state(self)->np.ndarray:
        h=w=self.world_size
        grid=np.zeros((h,w),dtype=np.uint8)
        wall=self.element_ids.get('wall',1)
        grid[0,:]=wall; grid[-1,:]=wall; grid[:,0]=wall; grid[:,-1]=wall
        candidates=[('sand',0.22),('water',0.18),('wood',0.14),('stone',0.10),('fire',0.08),('lava',0.05),('acid',0.06),('plant',0.10),('dust',0.07)]
        yy,xx=np.ogrid[:h,:w]
        for _ in range(int(self.rng.integers(5,10))):
            name,_=candidates[int(self.rng.integers(0,len(candidates)))]
            if name not in self.element_ids: continue
            elem=self.element_ids[name]
            if self.rng.random()<0.65:
                cy=int(self.rng.integers(4,h-4)); cx=int(self.rng.integers(4,w-4)); r=int(self.rng.integers(2,max(3,h//7)))
                mask=(yy-cy)**2+(xx-cx)**2<=r*r
            else:
                y0=int(self.rng.integers(2,h-8)); x0=int(self.rng.integers(2,w-8))
                y1=min(h-1,y0+int(self.rng.integers(3,max(4,h//4)))); x1=min(w-1,x0+int(self.rng.integers(3,max(4,w//4))))
                mask=np.zeros_like(grid,dtype=bool); mask[y0:y1,x0:x1]=True
            mask &= grid != wall
            grid[mask]=elem
        return grid
    def make_action_bank(self,k:int)->List[PWAction]:
        out=[]
        for _ in range(int(k)):
            elem=int(self.rng.choice(self.action_elements)); typ='erase' if elem==self.element_ids.get('empty',0) else 'place'
            out.append(PWAction(typ,elem,int(self.rng.integers(2,self.world_size-2)),int(self.rng.integers(2,self.world_size-2)),int(self.rng.choice([1,2,3],p=[.25,.55,.20]))))
        return out
    def apply_action(self,grid:np.ndarray,action:PWAction)->np.ndarray:
        g=np.asarray(grid,dtype=np.uint8).copy(); yy,xx=np.ogrid[:g.shape[0],:g.shape[1]]
        mask=(yy-int(action.y))**2+(xx-int(action.x))**2<=int(action.radius)**2; mask &= g!=self.element_ids.get('wall',1); g[mask]=int(action.element); return g
    def simulate(self,grid:np.ndarray,steps:int)->np.ndarray:
        torch = self.torch
        with torch.no_grad():
            if self._next_sim_seed is not None:
                torch.manual_seed(int(self._next_sim_seed))
            world=self.grid_to_world(grid)
            for _ in range(int(steps)): world=self.pw(world)
            return self.world_to_grid(world)
    def noop_rollout(self,grid:np.ndarray,horizon:int)->PWRollout:
        return PWRollout(self.simulate(grid,horizon),np.zeros(len(self.EVENT_NAMES),dtype=np.float32))
    def rollout(self,grid:np.ndarray,action:PWAction,horizon:int)->PWRollout:
        final=self.simulate(self.apply_action(grid,action),horizon); noop=self.simulate(grid,horizon); return PWRollout(final,self.event_vector(noop,final))
    def render_rgb(self,grid:np.ndarray)->np.ndarray:
        world=self.grid_to_world(grid); rgb=self.pwr.render(world).astype(np.float32)/255.0; return np.moveaxis(rgb,-1,0).astype(np.float32)
    def render_range(self,grid:np.ndarray)->np.ndarray:
        g=np.asarray(grid); h,w=g.shape; blockers=np.isin(g,[self.element_ids.get('wall',1),self.element_ids.get('stone',9),self.element_ids.get('wood',5)])
        out=np.zeros((4,h,w),dtype=np.float32); md=float(max(h,w))
        for x in range(w):
            last=-1
            for y in range(h):
                if blockers[y,x]: last=y; out[0,y,x]=0
                else: out[0,y,x]=(y-last)/md
            last=h
            for y in range(h-1,-1,-1):
                if blockers[y,x]: last=y; out[1,y,x]=0
                else: out[1,y,x]=(last-y)/md
        for y in range(h):
            last=-1
            for x in range(w):
                if blockers[y,x]: last=x; out[2,y,x]=0
                else: out[2,y,x]=(x-last)/md
            last=w
            for x in range(w-1,-1,-1):
                if blockers[y,x]: last=x; out[3,y,x]=0
                else: out[3,y,x]=(last-x)/md
        return np.clip(out,0,1)
    def render_local(self,grid:np.ndarray,action:PWAction,radius:int=3)->np.ndarray:
        g=np.asarray(grid,dtype=np.int32); wall=self.element_ids.get('wall',1); pad=np.pad(g,radius,mode='constant',constant_values=wall); y=int(action.y)+radius; x=int(action.x)+radius; patch=pad[y-radius:y+radius+1,x-radius:x+radius+1]
        elem=np.zeros_like(patch,dtype=np.float32); density=np.zeros_like(patch,dtype=np.float32); gravity=np.zeros_like(patch,dtype=np.float32)
        for idx,val in np.ndenumerate(patch):
            name=self.element_names.get(int(val),'empty'); tup=self.pw_elements.get(name,(0,0,0)); elem[idx]=float(tup[0])/max(1,float(len(self.pw_elements)-1)); density[idx]=float(tup[1])/4.0; gravity[idx]=float(tup[2])
        return np.stack([elem,density,gravity,(patch!=self.element_ids.get('empty',0)).astype(np.float32)],axis=0).astype(np.float32)
    def render_channel(self,grid:np.ndarray,channel:str,action:PWAction|None=None)->np.ndarray:
        if channel=='rgb': return self.render_rgb(grid)
        if channel=='range': return self.render_range(grid)
        if channel=='local':
            if action is None: raise ValueError('local channel requires action')
            return self.render_local(grid,action)
        if channel=='semantic': return np.asarray(grid,dtype=np.float32)[None]/max(1.0,float(len(self.pw_elements)-1))
        raise KeyError(f'Unsupported D0 channel {channel!r}')
    def event_vector(self,before:np.ndarray,after:np.ndarray)->np.ndarray:
        before=np.asarray(before,dtype=np.uint8); after=np.asarray(after,dtype=np.uint8); changed=before!=after; ids=self.element_ids
        def gain(n): return float(np.mean((after==ids.get(n,255))&changed))
        def loss(n): return float(np.mean((before==ids.get(n,255))&changed))
        return np.array([gain('fire'),loss('fire'),gain('water'),loss('water'),gain('wood'),loss('wood'),gain('lava'),gain('acid'),gain('plant')],dtype=np.float32)
