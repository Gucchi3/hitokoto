// @ts-check

// One sidebar for each tab: Docs (how to use hitokoto), Learn (tiny language models themselves), and API (one page per file of the code)

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docs: [
    {
      type: 'category',
      label: 'Get started',
      collapsed: false,
      items: ['intro', 'install', 'quickstart'],
    },
    'roadmap',
  ],
  learn: ['learn/index'],
  api: ['api/index'],
};

export default sidebars;
