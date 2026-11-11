// SPDX-FileCopyrightText: 2026 Deimantas Kiršinas
// SPDX-License-Identifier: Apache-2.0
// Minimali OGraf (v1) grafika: web komponentas su load / dispose / playAction / stopAction / updateAction.
// Sintetinė; skirta tik pavyzdžiui. Fonas visiškai permatomas, plokštė turi minkštą šešėlį (tarpinė alfa).
export default class LowerThird extends HTMLElement {
  async load(params) {
    const d = (params && params.data) || {};
    this.attachShadow({ mode: 'open' });
    this.shadowRoot.innerHTML = `
      <style>
        :host { position: absolute; inset: 0; display: block; }
        .plate { position: absolute; left: 120px; bottom: 140px; width: 1000px; height: 170px; border-radius: 28px;
          background: linear-gradient(90deg, rgba(0,70,160,.92), rgba(0,120,220,.55));
          box-shadow: 0 18px 60px rgba(0,0,0,.65); transform: translateX(-1300px); opacity: 0;
          transition: transform .8s cubic-bezier(.2,.8,.2,1), opacity .5s; }
        .plate.on { transform: none; opacity: 1; }
        h1 { margin: 26px 40px 0; font: 700 64px sans-serif; color: #fff; text-shadow: 0 2px 8px rgba(0,0,0,.6); }
        p { margin: 4px 40px; font: 400 36px sans-serif; color: rgba(255,255,255,.85); }
      </style>
      <div class="plate"><h1></h1><p></p></div>`;
    this.plate = this.shadowRoot.querySelector('.plate');
    await this.updateAction({ data: d });
  }
  async dispose() { this.shadowRoot.innerHTML = ''; }
  async updateAction({ data }) {
    if (data.title !== undefined) this.shadowRoot.querySelector('h1').textContent = data.title;
    if (data.subtitle !== undefined) this.shadowRoot.querySelector('p').textContent = data.subtitle;
  }
  async playAction() { this.plate.classList.add('on'); return { statusCode: 200, currentStep: 0 }; }
  async stopAction() { this.plate.classList.remove('on'); }
}
