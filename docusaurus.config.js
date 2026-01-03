const config = {
  title: 'Physical AI and Humanoid Robots',
  tagline: 'A simulation-first, hands-on textbook',
  favicon: 'img/favicon.ico',

  url: 'https://sufyanali92.github.io',
  baseUrl: '/book-_Hakathon/',
  organizationName: 'sufyanali92', // GitHub username
  projectName: 'book-_Hakathon',    // Corrected repo name
  deploymentBranch: 'gh-pages',

  trailingSlash: false,
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.js',
          editUrl:
            'https://github.com/sufyanali92/book-_Hakathon/edit/main/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: '',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Book',
        },
        {
          href: 'https://github.com/sufyanali92/book-_Hakathon',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [],
      copyright: '© 2026 Sufyan Ali', // <-- must not be empty
    },
    prism: {
      theme: require('prism-react-renderer/themes/github'),
      darkTheme: require('prism-react-renderer/themes/dracula'),
      additionalLanguages: ['python', 'bash', 'json', 'cpp'],
    },
  },
};

export default config;
